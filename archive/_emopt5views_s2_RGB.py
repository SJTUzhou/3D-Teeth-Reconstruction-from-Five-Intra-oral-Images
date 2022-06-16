import numpy as np
import pandas as pd
import open3d as o3d
from matplotlib import pyplot as plt
from shapely.geometry import MultiLineString, Point
from shapely.ops import unary_union, polygonize
import scipy
from scipy.spatial import Delaunay, distance_matrix
from scipy.spatial.transform import Rotation as RR
from collections import Counter
import itertools
import skimage
import enum

@enum.unique
class PHOTO(enum.Enum):
    # Enum values must be 0,1,2,3,4
    UPPER = 0
    LOWER = 1
    LEFT = 2
    RIGHT = 3
    FRONTAL = 4


class EMOpt5Views(object):
    # 使用COBYLA的启发式算法进行优化
    def __init__(self, edgeMasks, photoTypes, visMasks, Mask, Mu, SqrtEigVals, Sigma, PoseCovMats, ScaleCovMat, transVecStd, rotAngleStd) -> None:
        self.photoTypes = sorted(photoTypes, key=lambda x:x.value)
        assert self.photoTypes == [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
        
        self.edgeMaskU = [None] * 5 # order as the Enum value in PHOTO
        self.edgeMaskL = [None] * 5 # order as the Enum value in PHOTO
        self.visIdx = [None] * 5 # 每张照片中出现的牙齿轮廓的牙齿的Mask
        for phType, eMask, visMask in zip(photoTypes, edgeMasks, visMasks):
            assert eMask.ndim == 3, "edgeMask should be grayscale" # 单通道Mask图片
            self.edgeMaskU[phType.value] = eMask[...,0] # binary 2d-array red
            self.edgeMaskL[phType.value] = eMask[...,1] # binary 2d-array green
            self.visIdx[phType.value] = np.argwhere(visMask[Mask]>0).flatten()
        
        self.P_true_U = [np.argwhere(v>0)[:,::-1] for v in self.edgeMaskU] # real edge point pos in image coord (u_x,v_y), 2d-array, shape=(?,2)
        self.P_true_L = [np.argwhere(v>0)[:,::-1] for v in self.edgeMaskL]
        self.P_true = [np.vstack([pu,pl]) for pu,pl in zip(self.P_true_U,self.P_true_L)]
        self.M_U = [len(v) for v in self.P_true_U] # 真实Mask中边缘像素点的数量
        self.M_L = [len(v) for v in self.P_true_L]
        self.P_true_U_normals = [self.__initEdgeMaskNormals(v) if len(v)>0 else None for v in self.P_true_U]
        self.P_true_L_normals = [self.__initEdgeMaskNormals(v) if len(v)>0 else None for v in self.P_true_L]
        
        # 分为上下牙列
        Mask_U, Mask_L = np.split(Mask, 2, axis=0)
        self.numUpperTooth = int(np.sum(Mask_U)) #上牙列的牙齿数量
        self.numTooth = int(np.sum(Mask))
        self.numPoint = Mu.shape[1]
        # 记录正视图、左视图、右视图中上下牙列visIdx的区分id
        self.ul_sp = {phType.value:np.argwhere(self.visIdx[phType.value] >= self.numUpperTooth).min() for phType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]}

        # 上牙列index=0 下牙列index=1
        self.X_Mu = Mu[Mask]
        self.X_Mu_centroids = self.X_Mu.mean(axis=1)
        # self.X_Mu_U_Cen =  self.X_Mu_centroids[:self.numUpperTooth].mean(axis=0) # 原点 in world coord
        # self.X_Mu_L_Cen =  self.X_Mu_centroids[self.numUpperTooth:].mean(axis=0) # 原点 in world coord

        self.X_Mu_normals = self.computePointNormals(self.X_Mu)

        self.SqrtEigVals = SqrtEigVals[Mask]
        self.SigmaT = np.transpose(Sigma[Mask],(0,2,1))

        self.meanRotAngleXYZs = np.zeros((self.numTooth,3)) # 每颗牙齿相对旋转角度的均值 # 待讨论
        self.meanTransVecXYZs = np.zeros((self.numTooth,3)) # 每颗牙齿相对平移的均值 # 待讨论
        self.meanScales = np.ones((self.numTooth,)) #每颗牙齿的相对尺寸
        self.invCovMats = np.linalg.inv(PoseCovMats[Mask])
        self.invCovMatOfScale = np.linalg.inv(ScaleCovMat[Mask][:,Mask])

        # init teeth shape subspace
        self.numPC = SqrtEigVals.shape[-1] 
        self.featureVec = np.zeros(self.SqrtEigVals.shape, dtype=np.float32) # shape=(self.numTooth, 1, numPC), mean=0, std=1
        
        # init teeth scales, rotation angles around X-Y-Z axes, translation vectors along X-Y-Z axes
        self.scales = np.ones((self.numTooth,))
        self.rotAngleXYZs = np.zeros((self.numTooth,3))
        self.transVecXYZs = np.zeros((self.numTooth,3))
        
        self.rowScaleXZ = np.ones((2,), dtype=np.float32) # 各向异性牙列放缩, 仅用于maximization stage1, 在stage2之前转化为scale和transVecXYZs

        # init extrinsic param of camera
        self.ex_rxyz = np.vstack([self.__initExtrinsicRotAngles(k) for k in self.photoTypes]) # shape=(5,3) # init rot angles around x-y-z axis based on photoType
        obj_dist_estimates = [45.,45.,35.,35.,35.]
        self.ex_txyz = np.vstack([self.__initExtrinsicTransVec(k, obj_dist_estimate=obj_d_est)\
             for k,obj_d_est in zip(self.photoTypes,obj_dist_estimates)]) # shape=(5,3) # init trans vector
        self.rela_rxyz = np.array([0.,0.,0.],dtype=np.float32) #下牙列相对于上牙列的旋转
        self.rela_R = self.updateRelaRotMat(self.rela_rxyz)
        self.rela_txyz = np.array([0.,-5.,0.],dtype=np.float32) #下牙列相对于上牙列的位移

        self.ex_rxyz_lr = 1.
        self.ex_txyz_lr = 1.
        self.focLth_lr = 1.
        self.uv_lr = 1.
        self.dpix_lr = 1.
        self.rela_rxyz_lr = 1.
        self.rela_txyz_lr = 1.
        
        # init intrinsic param of camera
        self.focLth =  np.array([50.0,50.0,35.0,35.0,35.0],dtype=np.float32)
        self.dpix = np.array([0.06,0.06,0.06,0.06,0.06],dtype=np.float32)
        self.u0 = np.array([v.shape[1]//2 for v in self.edgeMaskU], dtype=np.uint32) # img.width//2
        self.v0 = np.array([v.shape[0]//2 for v in self.edgeMaskU], dtype=np.uint32) # img.height//2
        
        self.varAngle = 0.09 # param in expectation loss
        
        self.varPoint = 25.  # param in residual pixel error in maximization loss
        self.varPlane = 0.5  # param in residual pixel error in maximization loss
        self.weightViewMaxiStage1 = np.array([3.,3.,1.,1.,1.], dtype=np.float32)
        self.weightViewMaxiStage2 = np.array([3.,3.,1.,1.,1.], dtype=np.float32) # weight in maximization step for 5 views: [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
        self.weightAniScale = 1.
        self.weightTeethPose = 1. # param in residual teeth pose error in maximization loss
        self.weightFeatureVec = 1. # param in residual featureVec error in maximization loss
        
        # 用于设置COBYLA优化过程中的步长rhobeg
        self.transVecStd = transVecStd
        self.scaleStd = np.mean(np.sqrt(np.diag(ScaleCovMat)))
        self.rotAngleStd = rotAngleStd


        self.X_deformed = np.empty(self.X_Mu.shape, dtype=np.float32)
        self.X_deformed_normals = np.empty(self.X_Mu.shape, dtype=np.float32)
        self.RotMats = np.empty((self.numTooth,3,3), dtype=np.float32)
        self.X_trans = np.empty(self.X_Mu.shape, dtype=np.float32)
        self.X_trans_normals = np.empty(self.X_Mu.shape, dtype=np.float32)
        self.updateAlignedPointCloudInWorldCoord(self.visIdx[PHOTO.UPPER.value]) # 更新上下牙列世界坐标系中的三维预测
        self.updateAlignedPointCloudInWorldCoord(self.visIdx[PHOTO.LOWER.value])

        self.extrViewMat = np.empty((5,4,3), dtype=np.float32) # homo world coord (xw,yw,zw,1) to camera coord (xc,yc,zc): 4*3 right-multiplying matrix
        self.X_camera = [None] * 5 # compute X in camera coord based on X in world coord, ndarray, shape=(numTooth,1500,3)
        self.X_camera_normals = [None] * 5
        
        self.intrProjMat = np.empty((5,3,3), dtype=np.float32) # camera coord (xc,yc,zc) to image coord (u,v,zc): 3*3 right-multiplying matrix
        self.X_uv = [None] * 5 # compute X in image coord based on X_camera in camera coord, ndarray, shape=(numTooth,1500,2)
        self.X_uv_normals = [None] * 5
        
        self.vis_hull_vertices = [None] * 5
        self.vis_hull_vertex_indices = [None] * 5 # visible points in image coord, and corre idx in X
        self.P_pred_U = [None] * 5 # edgeMask prediction 2d-array, shape=(?,2)
        self.P_pred_L = [None] * 5
        self.P_pred_U_normals = [None] * 5
        self.P_pred_L_normals = [None] * 5
        self.X_Mu_pred = [None] * 5 # P_pred 对应的原始点云中的点
        self.X_Mu_pred_normals = [None] * 5 # P_pred 对应的原始点云中的点的法向量
        self.X_deformed_pred = [None] * 5
        self.X_deformed_pred_normals = [None] * 5

        self.SigmaT_segs = [None] * 5

        for phType in self.photoTypes:
            self.updateEdgePrediction(phType)

        self.loss_expectation_step = [None] * 5
        self.corre_pred_idx = [None] * 5
        self.loss_maximization_step = 0.

    def computePointNormals(self, X):
        # X.shape=(self.numTooth,self.numPoint,3)
        # 分别计算X中每组点云的法向量
        normals = []
        for vertices in X:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            pcd.estimate_normals()
            # to obtain a consistent normal orientation
            pcd.orient_normals_consistent_tangent_plane(k=15)
            pcd.normalize_normals()
            normals.append(np.asarray(pcd.normals,dtype=np.float32))
        return np.array(normals,dtype=np.float32)



    ###########################################
    ######### Initialization functions ########
    ###########################################

    def __initEdgeMaskNormals(self, vertices_xy):
        # 计算edgeMask ground truth中边缘点的法向量, shape = (M,2)
        M = len(vertices_xy)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.hstack([vertices_xy, 20*np.random.rand(M,1)]))
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=30)
        normals_xy = np.asarray(pcd.normals)[:,:2]
        pcd.normals = o3d.utility.Vector3dVector(np.hstack([normals_xy, np.zeros((M,1))]))
        pcd.normalize_normals()
        # o3d.visualization.draw_geometries([pcd], window_name="image edge normals estimation", width=800, height=600, left=50,top=50, point_show_normal=True)
        return np.asarray(pcd.normals, dtype=np.float32)[:,:2]

    def __initExtrinsicRotAngles(self, photoType):
        if photoType == PHOTO.UPPER:
            return np.array([-0.3*np.pi, np.pi, np.pi], dtype=np.float32) # upper
        elif photoType == PHOTO.LOWER:
            return np.array([0.3*np.pi, 0.95*np.pi, np.pi], dtype=np.float32)  # lower
        elif photoType == PHOTO.LEFT:
            return np.array([0.0, 0.8*np.pi, np.pi], dtype=np.float32) # left
        elif photoType == PHOTO.RIGHT:
            return np.array([0.0, -0.8*np.pi, np.pi], dtype=np.float32)  # right
        elif photoType == PHOTO.FRONTAL:
            return np.array([0.0, np.pi, np.pi], dtype=np.float32)  # frontal
        else:
            assert photoType in PHOTO, "photoType should be a PHOTO Enum"
    
    def __initExtrinsicTransVec(self, photoType, obj_dist_estimate):
        # obj_dist_estimate: 物距 mm
        if photoType == PHOTO.UPPER:
            return np.array([0., 0., -self.X_Mu[...,2].min()+obj_dist_estimate], dtype=np.float32) # upper
        elif photoType == PHOTO.LOWER:
            return np.array([0., 0., -self.X_Mu[...,2].min()+obj_dist_estimate], dtype=np.float32)  # lower
        elif photoType == PHOTO.LEFT:
            return np.array([-5., 0., -self.X_Mu[...,2].min()+obj_dist_estimate], dtype=np.float32) # left
        elif photoType == PHOTO.RIGHT:
            return np.array([5., 0., -self.X_Mu[...,2].min()+obj_dist_estimate], dtype=np.float32)  # right
        elif photoType == PHOTO.FRONTAL:
            return np.array([0., 0., -self.X_Mu[...,2].min()+obj_dist_estimate], dtype=np.float32)  # frontal
        else:
            assert photoType in PHOTO, "photoType should be a PHOTO Enum"



    ###############################################
    # Deformatin in shape subspace for each tooth #
    ###############################################

    def updateDeformedPointPos(self, featureVec, tIdx):
        deformField = np.matmul(featureVec*self.SqrtEigVals[tIdx], self.SigmaT[tIdx]) # shape=(numTooth,1,3*self.numPoint)
        return self.X_Mu[tIdx] + deformField.reshape(self.X_Mu[tIdx].shape) # X_deformed

    def updateDeformedPointNomrals(self):
        pass


    ########################################################
    # Isotropic scaled rigid transformation for each tooth #
    ########################################################
    def computeTeethRotMats(self, rotAngleXYZs):
        rotMats = RR.from_euler("xyz", rotAngleXYZs).as_matrix()
        rotMats = np.transpose(rotMats, (0,2,1)) # 变为右乘
        return rotMats
    
    def updateTransformedPointPos(self, X_deformed, scales, rotMats, transVecXYZs, tIdx):
        # X_trans = scales_inv * (X_deformed + transVecXYZs_inv) @ rotMats_inv
        # in CPD: X_aligned_deformed = scales_cpd * X_trans @ rotMats_cpd + transVecXYZs_cpd
        return np.multiply(scales[:,None,None], np.matmul(X_deformed-self.X_Mu_centroids[tIdx,None,:], rotMats)) +\
                transVecXYZs[:,None,:] + self.X_Mu_centroids[tIdx,None,:]
    
    def updateTransformedPointNormals(self, X_deformed_normals, rotMats):
        # 法向量只需要计算旋转即可
        return np.matmul(X_deformed_normals, rotMats)


    ####################################################################
    # Relative Pose of Lower Tooth Row with respect to Upper Tooth Row #
    ####################################################################

    def updateRelaRotMat(self, rela_rxyz):
        return RR.from_euler("xyz", rela_rxyz).as_matrix().T

    def updateLowerPointPosByRelaPose(self, X_lower, rela_R, rela_txyz):
        return np.matmul(X_lower, rela_R) + rela_txyz

    def updateLowerPointNormalsByRelaPose(self, X_lower_normals, rela_R):
        return np.matmul(X_lower_normals, rela_R)


    ###############################
    # world coord -> camera coord #
    ###############################
    def updateExtrinsicViewMatrix(self, ex_rxyz, ex_txyz): # world coord to camera coord
        # 先进行x轴旋转，再y轴，再z轴；取转置表示右乘旋转矩阵，再平移
        R = RR.from_euler("xyz", ex_rxyz).as_matrix().T
        return np.vstack([R, ex_txyz]) # Matrix 4*3
        
    def updatePointPosInCameraCoord(self, X_world, extrViewMat):
        # get 3D point cloud in camera coord, return array shape (n,3) or (batch,n,3)
        X_homo = np.concatenate([X_world, np.ones((*X_world.shape[:-1],1))], axis=-1)
        return np.matmul(X_homo, extrViewMat)
    
    def updatePointNormalsInCameraCoord(self, X_world_normals, extrViewRotMat):
        return np.matmul(X_world_normals, extrViewRotMat)


    
    
    ##############################
    # camera coord ->image coord #
    ##############################
    
    def updateIntrinsicProjectionMatrix(self, focLth, dpix, u0, v0): # camera cood to image coord
        # mat1 = np.diag([focLth, focLth, 1.])
        # mat2 = np.array([[1./dpix, 0., 0.], [0., 1./dpix, 0.], [u0, v0, 1.]])
        # return mat1 @ mat2 # Matrix 3*3
        return np.array([[focLth/dpix, 0., 0.], [0., focLth/dpix, 0.], [u0, v0, 1.]])

    def updatePointPosInImageCoord(self, X_camera, intrProjMat):
        # get 3D point cloud in image coord, return array shape (n,2) or (batch,n,2)
        assert (X_camera[...,2]>0).all() # Z-value of points should be positive
        X_image = np.matmul((X_camera/X_camera[...,[2]]), intrProjMat)
        X_uv = X_image[...,:2]
        return np.around(X_uv).astype(np.int32)
    
    def updatePointNormalsInImageCoord(self, X_camera_normals):
        X_cam_normals_xy = X_camera_normals[...,:2] # 取相机坐标系下normals的x,y坐标即为图片坐标系中的normals
        return X_cam_normals_xy / np.linalg.norm(X_cam_normals_xy, axis=-1, keepdims=True)


    ##################################################
    # Extract contour pixels in projected pointcloud #
    ##################################################
    
    def __getUniquePixels(self, X_uv_int):
        # merge points at the same position in image coord
        # X_uv_int: array shape (n,2)dtype np.int32
        # pixels: array (m,2), each element represents (u_x, v_y)
        pixels, unique_indices = np.unique(X_uv_int,axis=0,return_index=True)
        return pixels, unique_indices
    
    def __getConcaveHullEdgeVertexIndices(self, coords, alpha):  # coords is a 2D numpy array (u_x,v_y)
        tri = Delaunay(coords, qhull_options="Qt Qc Qz Q12").simplices
        ia, ib, ic = (tri[:, 0], tri[:, 1], tri[:, 2])  # indices of each of the triangles' points
        pa, pb, pc = (coords[ia], coords[ib], coords[ic])  # coordinates of each of the triangles' points
        a = np.linalg.norm(pa-pb, ord=2, axis=1)
        b = np.linalg.norm(pb-pc, ord=2, axis=1)
        c = np.linalg.norm(pc-pa, ord=2, axis=1)
        s = (a + b + c) * 0.5  # Semi-perimeter of triangle
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Area of triangle by Heron's formula
        filter = (a * b * c / (4.0 * area) < 1.0 / alpha)  # Radius Filter based on alpha value
        edges = tri[filter]
        edges = [tuple(sorted(combo)) for e in edges for combo in itertools.combinations(e, 2)]
        count = Counter(edges)  # count occurrences of each edge
        edge_indices = [e for e, c in count.items() if c == 1]
        return np.array(edge_indices)
    
    def __constructConcaveHull(self, coords, edge_indices): # coords is a 2D numpy array (u_x,v_y)
        edges = [(coords[e[0]], coords[e[1]]) for e in edge_indices]
        ml = MultiLineString(edges)
        poly = polygonize(ml)
        hull = unary_union(list(poly))
        return hull
    
    def extractVisibleEdgePointsByAvgDepth(self, photoType):
        # X_uv: shape=(numTooth,numPoint,2), dtype=np.int32
        ph = photoType.value
        avg_depth = self.X_camera[ph][...,2].mean(axis=1) # avg_depth: array shape (numTooth,)
        tooth_order = avg_depth.argsort()
        X_uv_sort = self.X_uv[ph][tooth_order]
        hulls = []
        vis_hull_vs = []
        vis_hull_vids = []
        for x_uv in X_uv_sort:
            pixels, pixel_xuv_map = self.__getUniquePixels(x_uv)
            edge_v_indices = self.__getConcaveHullEdgeVertexIndices(pixels, alpha=0.05)
            hull = self.__constructConcaveHull(pixels, edge_v_indices)
            uni_edge_v_indices = np.unique(edge_v_indices)
            hull_v = x_uv[pixel_xuv_map[uni_edge_v_indices]]
            flags = np.ones((len(hull_v),),dtype=np.bool_)
            for i,v in enumerate(hull_v):
                for exist_hull in hulls:
                    if exist_hull.contains(Point(v)):
                        flags[i] = False
                        break
            if flags.any()==True:
                hulls.append(hull)
                vis_hull_vs.append(hull_v[flags]) # 可见点
                vis_hull_vids.append(pixel_xuv_map[uni_edge_v_indices[flags]])
        # sort in the init order
        vis_hull_vs = [x for _, x in sorted(zip(tooth_order, vis_hull_vs), key=lambda pair: pair[0])]
        vis_hull_vids = [x for _, x in sorted(zip(tooth_order, vis_hull_vids), key=lambda pair: pair[0])]
        return vis_hull_vs, vis_hull_vids




    ###########################################
    ######### Update in E step ################
    ###########################################
    def updateAlignedPointCloudInWorldCoord(self, tIdx):
        # 暂未考虑下牙列相对于上牙列的位姿关系
        self.X_deformed[tIdx] = self.updateDeformedPointPos(self.featureVec[tIdx], tIdx)
        self.X_deformed_normals[tIdx] = self.computePointNormals(self.X_deformed[tIdx])

        self.RotMats[tIdx] = self.computeTeethRotMats(self.rotAngleXYZs[tIdx])
        self.X_trans[tIdx] = self.updateTransformedPointPos(self.X_deformed[tIdx], self.scales[tIdx], self.RotMats[tIdx], self.transVecXYZs[tIdx], tIdx)
        self.X_trans[tIdx] = np.hstack([self.rowScaleXZ[0],1.,self.rowScaleXZ[1]]) * self.X_trans[tIdx] # self.rowScaleXZ = [1.,1.,1.] after maximization stage 1
        self.X_trans_normals[tIdx] = self.updateTransformedPointNormals(self.X_deformed_normals[tIdx], self.RotMats[tIdx])

    def updateEdgePrediction(self, photoType):
        # 根据拍摄角度，决定使用上牙列或下牙列点云
        ph = photoType.value
        tIdx = self.visIdx[ph]
        X_trans = self.X_trans[tIdx] # upper
        X_trans_normals = self.X_trans_normals[tIdx]

        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            self.rela_R = self.updateRelaRotMat(self.rela_rxyz)
            X_trans = np.concatenate([X_trans[:ul_sp], \
                self.updateLowerPointPosByRelaPose(X_trans[ul_sp:], self.rela_R, self.rela_txyz)], axis=0) # left, right, frontal
            X_trans_normals = np.concatenate([X_trans_normals[:ul_sp], \
                self.updateLowerPointNormalsByRelaPose(X_trans_normals[ul_sp:], self.rela_R)], axis=0)
        
        self.extrViewMat[ph] = self.updateExtrinsicViewMatrix(self.ex_rxyz[ph], self.ex_txyz[ph]) # homo world coord (xw,yw,zw,1) to camera coord (xc,yc,zc): 4*3 right-multiplying matrix
        self.X_camera[ph] = self.updatePointPosInCameraCoord(X_trans, self.extrViewMat[ph]) # compute X in camera coord based on X in world coord, ndarray, shape=(numTooth,1500,3)
        self.X_camera_normals[ph] = self.updatePointNormalsInCameraCoord(X_trans_normals, self.extrViewMat[ph,:3,:])
        
        self.intrProjMat[ph] = self.updateIntrinsicProjectionMatrix(self.focLth[ph], self.dpix[ph], self.u0[ph], self.v0[ph]) # camera coord (xc,yc,zc) to image coord (u,v,zc): 3*3 right-multiplying matrix
        self.X_uv[ph] = self.updatePointPosInImageCoord(self.X_camera[ph], self.intrProjMat[ph]) # compute X in image coord based on X_camera in camera coord, ndarray, shape=(numTooth,1500,2)
        self.X_uv_normals[ph] = self.updatePointNormalsInImageCoord(self.X_camera_normals[ph])
        
        self.vis_hull_vertices[ph], self.vis_hull_vertex_indices[ph] = self.extractVisibleEdgePointsByAvgDepth(photoType) # visible points in image coord, and corre idx in X
        if photoType == PHOTO.UPPER:
            self.P_pred_U[ph] = np.vstack(self.vis_hull_vertices[ph]) # edgeMask prediction 2d-array, shape=(?,2)
            self.P_pred_U_normals[ph] = np.vstack([x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_uv_normals[ph], self.vis_hull_vertex_indices[ph])])  # edgeMask normals prediction 2d-array, shape=(?,2)
        elif photoType == PHOTO.LOWER:
            self.P_pred_L[ph] = np.vstack(self.vis_hull_vertices[ph])
            self.P_pred_L_normals[ph] = np.vstack([x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_uv_normals[ph], self.vis_hull_vertex_indices[ph])])  # edgeMask normals prediction 2d-array, shape=(?,2)
        else:
            ul_sp = self.ul_sp[ph]
            self.P_pred_U[ph] = np.vstack(self.vis_hull_vertices[ph][:ul_sp]) 
            self.P_pred_U_normals[ph] = np.vstack([x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_uv_normals[ph][:ul_sp], self.vis_hull_vertex_indices[ph][:ul_sp])])
            self.P_pred_L[ph] = np.vstack(self.vis_hull_vertices[ph][ul_sp:]) 
            self.P_pred_L_normals[ph] = np.vstack([x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_uv_normals[ph][ul_sp:], self.vis_hull_vertex_indices[ph][ul_sp:])])
        
        
        self.X_Mu_pred[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_Mu[tIdx], self.vis_hull_vertex_indices[ph])] # points in world coord corre to edgeMask prediction
        self.X_Mu_pred_normals[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_Mu_normals[tIdx], self.vis_hull_vertex_indices[ph])]
        self.X_deformed_pred[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_deformed[tIdx], self.vis_hull_vertex_indices[ph])] 
        self.X_deformed_pred_normals[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_deformed_normals[tIdx], self.vis_hull_vertex_indices[ph])] 
        
    
    
    ###########################################
    ######### Update & Expectation Step #######
    ###########################################

    def expectation_step_5Views(self, verbose=True):
        # 根据新的edgePredcition计算对应点对关系,对5张图同时进行
        tIdx = [i for i in range(self.numTooth)]
        self.updateAlignedPointCloudInWorldCoord(tIdx)
        for photoType in self.photoTypes:
            ph = photoType.value
            self.updateEdgePrediction(photoType)
            losses = None
            if photoType == PHOTO.UPPER:
                point_loss_mat = distance_matrix(self.P_true_U[ph], self.P_pred_U[ph], p=2, threshold=int(1e8))**2
                normal_loss_mat = - (self.P_true_U_normals[ph] @ self.P_pred_U_normals[ph].T)**2 / self.varAngle
                loss_mat = point_loss_mat * np.exp(normal_loss_mat)
                self.corre_pred_idx[ph] = np.argmin(loss_mat, axis=1)
                losses = loss_mat[np.arange(self.M_U[ph]), self.corre_pred_idx[ph]]
            elif photoType == PHOTO.LOWER:
                point_loss_mat = distance_matrix(self.P_true_L[ph], self.P_pred_L[ph], p=2, threshold=int(1e8))**2
                normal_loss_mat = - (self.P_true_L_normals[ph] @ self.P_pred_L_normals[ph].T)**2 / self.varAngle
                loss_mat = point_loss_mat * np.exp(normal_loss_mat)
                self.corre_pred_idx[ph] = np.argmin(loss_mat, axis=1)
                losses = loss_mat[np.arange(self.M_L[ph]), self.corre_pred_idx[ph]]
            else:
                point_loss_mat_U = distance_matrix(self.P_true_U[ph], self.P_pred_U[ph], p=2, threshold=int(1e8))**2
                normal_loss_mat_U = - (self.P_true_U_normals[ph] @ self.P_pred_U_normals[ph].T)**2 / self.varAngle
                point_loss_mat_L = distance_matrix(self.P_true_L[ph], self.P_pred_L[ph], p=2, threshold=int(1e8))**2
                normal_loss_mat_L = - (self.P_true_L_normals[ph] @ self.P_pred_L_normals[ph].T)**2 / self.varAngle
                loss_mat_U = point_loss_mat_U * np.exp(normal_loss_mat_U)
                loss_mat_L = point_loss_mat_L * np.exp(normal_loss_mat_L)
                _num_u = loss_mat_U.shape[1]
                self.corre_pred_idx[ph] = np.hstack([np.argmin(loss_mat_U, axis=1), _num_u+np.argmin(loss_mat_L, axis=1)])
                losses = np.hstack([loss_mat_U[np.arange(self.M_U[ph]), self.corre_pred_idx[ph][:self.M_U[ph]]], \
                    loss_mat_L[np.arange(self.M_L[ph]), self.corre_pred_idx[ph][self.M_U[ph]:]-_num_u] ])
            self.loss_expectation_step[ph] = np.sum(losses)
            if verbose==True:
                print("{} - unique pred points: {} - E-step loss: {:.2f}".format(str(photoType), len(np.unique(self.corre_pred_idx[ph])), self.loss_expectation_step[ph]))

    ###########################################
    ######### Maximization Step ###############
    ###########################################
    def computePixelResidualError(self, photoType, featureVec, scales, rotAngleXYZs, transVecXYZs, extrViewMat, intrProjMat,\
        rela_R, rela_txyz, rowScaleXZ=np.ones((2,),np.float32), stage=1):
        # self.X_?_pred: List of array of points in Mu teeth shape, [ndarray1, ndarray2, ...]
        # self.corre_pred_idx: corre indices after vertically stacking the transformed self.X_?_pred
        
        ph = photoType.value
        tIdx = self.visIdx[ph]
        X_deformed_pred = self.X_Mu_pred[ph]
        X_trans_pred = self.X_Mu_pred[ph]
        X_deformed_pred_normals = self.X_Mu_pred_normals[ph]
        X_trans_pred_normals = self.X_Mu_pred_normals[ph]


        if stage >= 3: # 考虑shape subspace 的形变
            # X_deformed = self.updateDeformedPointPos(featureVec, tIdx) # ul may be 0,1,-1 # 轮廓点对应原始点云进行Shape subspace变形操作
            # X_deformed_normals = self.computePointNormals(X_deformed)
            # X_deformed_pred = [x[vis_hull_vids] for x,vis_hull_vids in zip(X_deformed, self.vis_hull_vertex_indices[ph])]
            # X_deformed_pred_normals = [x[vis_hull_vids] for x,vis_hull_vids in zip(X_deformed_normals, self.vis_hull_vertex_indices)]
            X_deformed_pred = [x_mu_pred+np.reshape(sqrtEigVal*fVec@sigmaTseg,x_mu_pred.shape) for x_mu_pred,sqrtEigVal,fVec,sigmaTseg in \
                              zip(self.X_Mu_pred[ph], self.SqrtEigVals[tIdx], featureVec, self.SigmaT_segs[ph])] # 轮廓点对应原始点云进行Shape subspace变形操作
            X_deformed_pred_normals = self.X_deformed_pred_normals[ph]
        
        if stage >= 2: # 考虑每颗牙齿的相对位姿和尺寸
            rotMats = self.computeTeethRotMats(rotAngleXYZs)
            X_trans_pred = [s*np.matmul(x-tc,R)+t+tc for x,s,R,t,tc in zip(X_deformed_pred, scales, rotMats, transVecXYZs, self.X_Mu_centroids[tIdx])] # 轮廓点对应原始点云按牙齿分别进行缩放刚性变换
            X_trans_pred_normals = [np.matmul(xn,R) for xn,R in zip(X_deformed_pred_normals, rotMats)]
            
        
        # 需要考虑上下牙列位置关系，对下牙列的点进行相对旋转和平移
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            X_trans_pred = X_trans_pred[:ul_sp] + [x@rela_R+rela_txyz for x in X_trans_pred[ul_sp:]]
            X_trans_pred_normals = X_trans_pred_normals[:ul_sp] + [xn@rela_R for xn in X_trans_pred_normals[ul_sp:]]
        
        X_corre_pred = np.vstack(X_trans_pred)[self.corre_pred_idx[ph]]            
        if stage == 1: # 在优化相机参数同时，优化牙列的Anistropic scales
            X_corre_pred = np.hstack([rowScaleXZ[0],1.,rowScaleXZ[1]]) * X_corre_pred
        X_corre_pred_normals = np.vstack(X_trans_pred_normals)[self.corre_pred_idx[ph]]

        X_cam_corre_pred = self.updatePointPosInCameraCoord(X_corre_pred, extrViewMat) #相机坐标系下对应点坐标
        X_cam_corre_pred_normals = self.updatePointNormalsInCameraCoord(X_corre_pred_normals, extrViewMat[:3,:]) # extrViewMat.shape = (4,3)
        
        P_corre_pred = self.updatePointPosInImageCoord(X_cam_corre_pred, intrProjMat)
        P_corre_pred_normals = self.updatePointNormalsInImageCoord(X_cam_corre_pred_normals)
        
        errorVecUV = self.P_true[ph] - P_corre_pred # ci - \hat{ci}
        resPointError = np.sum(np.linalg.norm(errorVecUV, axis=1)**2) / self.varPoint
        resPlaneError = np.sum(np.sum(errorVecUV*P_corre_pred_normals, axis=1)**2) / self.varPlane
        return (resPointError + resPlaneError) / (self.M_U[ph] + self.M_L[ph])

    def parseGlobalParamsOfSingleView(self, params, pIdx, photoType):
        ex_rxyz = self.ex_rxyz_lr * params[pIdx["ex_rxyz"]:pIdx["ex_rxyz"]+3]
        ex_txyz = self.ex_txyz_lr * params[pIdx["ex_txyz"]:pIdx["ex_txyz"]+3]
        focLth = self.focLth_lr * params[pIdx["focLth"]]
        dpix = self.dpix_lr * params[pIdx["dpix"]]
        u0, v0 = self.uv_lr * params[pIdx["u0v0"]:pIdx["u0v0"]+2]
        rela_rxyz = self.rela_rxyz
        rela_txyz = self.rela_txyz
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            rela_rxyz = self.rela_rxyz_lr * params[pIdx["rela_rxyz"]:pIdx["rela_rxyz"]+3]
            rela_txyz = self.rela_txyz_lr * params[pIdx["rela_txyz"]:pIdx["rela_txyz"]+3]
        return ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz

    def parseGlobalParamsOf5Views(self, params, pIdx):
        ex_rxyz = self.ex_rxyz_lr * params[pIdx["ex_rxyz"]:pIdx["ex_rxyz"]+15].reshape(5,3)
        ex_txyz = self.ex_txyz_lr * params[pIdx["ex_txyz"]:pIdx["ex_txyz"]+15].reshape(5,3)
        focLth = self.focLth_lr * params[pIdx["focLth"]:pIdx["focLth"]+5]
        dpix = self.dpix_lr * params[pIdx["dpix"]:pIdx["dpix"]+5]
        u0 = self.uv_lr * params[pIdx["u0"]:pIdx["u0"]+5]
        v0 = self.uv_lr * params[pIdx["v0"]:pIdx["v0"]+5]
        rela_rxyz = self.rela_rxyz_lr * params[pIdx["rela_rxyz"]:pIdx["rela_rxyz"]+3]
        rela_txyz = self.rela_txyz_lr * params[pIdx["rela_txyz"]:pIdx["rela_txyz"]+3]
        return ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz

    def parseTeethPoseParams(self, params, pIdx, tIdx, step):
        transVecXYZs = self.transVecXYZs[tIdx]
        rotAngleXYZs = self.rotAngleXYZs[tIdx]
        scales = self.scales[tIdx]
        numT = len(tIdx)
        if step == 1:
            transVecXYZs = params[pIdx["tXYZs"]:pIdx["tXYZs"]+numT*3].reshape(numT, 3)
        elif step == 2:
            rotAngleXYZs = params[pIdx["rXYZs"]:pIdx["rXYZs"]+numT*3].reshape(numT, 3)
        elif step == 3:
            scales = params[pIdx["scales"]:pIdx["scales"]+numT]
        elif step == 4:
            transVecXYZs = self.transVecStd * params[pIdx["tXYZs"]:pIdx["tXYZs"]+numT*3].reshape(numT, 3)
            rotAngleXYZs = self.rotAngleStd * params[pIdx["rXYZs"]:pIdx["rXYZs"]+numT*3].reshape(numT, 3)
            scales = 1. + self.scaleStd * params[pIdx["scales"]:pIdx["scales"]+numT]
        return transVecXYZs, rotAngleXYZs, scales

    def getGlobalParamsOfSingleView_as_x0(self, photoType, ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz):
        # 生成 scipy.optimize.minimize 函数的部分输入
        ph = photoType.value
        pIdx = {"ex_rxyz":0, "ex_txyz":3, "focLth":6, "dpix":7, "u0v0":8} # 与x0相对应
        x0 = np.hstack([ex_rxyz[ph]/self.ex_rxyz_lr, ex_txyz[ph]/self.ex_txyz_lr, \
            focLth[ph]/self.focLth_lr, dpix[ph]/self.dpix_lr, u0[ph]/self.uv_lr, v0[ph]/self.uv_lr])
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            pIdx.update({"rela_rxyz":len(x0), "rela_txyz":len(x0)+3})
            x0 = np.hstack([x0, rela_rxyz/self.rela_rxyz_lr, rela_txyz/self.rela_txyz_lr])
        return x0, pIdx

    def getCurrentGlobalParamsOf5Views_as_x0(self):
        pIdx = {"ex_rxyz":0, "ex_txyz":15, "focLth":30, "dpix":35, "u0":40, "v0":45,\
            "rela_rxyz":50, "rela_txyz":53} # 与x0相对应
        x0 = np.hstack([self.ex_rxyz.flatten()/self.ex_rxyz_lr, self.ex_txyz.flatten()/self.ex_txyz_lr, \
            self.focLth/self.focLth_lr, self.dpix/self.dpix_lr, self.u0/self.uv_lr, self.v0/self.uv_lr, \
            self.rela_rxyz/self.rela_rxyz_lr, self.rela_txyz/self.rela_txyz_lr])
        return x0, pIdx

    ###########################################
    ######### Maximization stage 1 ############
    ###########################################

    def residualErrorOfGlobalParams(self, params, pIdx, photoType, withAniScales=False, verbose=True):
        # params = np.array([ex_rx, ex_ry, ex_rz, ex_tx, ex_ty, ex_tz, f, u0, v0])
        ph = photoType.value
        ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz = self.parseGlobalParamsOfSingleView(params, pIdx, photoType)
        rowScaleXZ = np.array([1.,1.], np.float32)
        errorAniScales = 0.
        if withAniScales == True: # 考虑牙列各向异性放缩
            rowScaleXZ = params[pIdx["rowScaleXZ"]:pIdx["rowScaleXZ"]+2] 
            errorAniScales = self.weightAniScale * np.sum((rowScaleXZ - 1.)**2) / self.scaleStd**2
        
        rela_R = self.updateRelaRotMat(rela_rxyz)
        extrViewMat = self.updateExtrinsicViewMatrix(ex_rxyz, ex_txyz)
        intrProjMat = self.updateIntrinsicProjectionMatrix(focLth, dpix, u0, v0)
        tIdx = self.visIdx[ph]
        errorPixel = self.computePixelResidualError(photoType, self.featureVec[tIdx], self.scales[tIdx], self.rotAngleXYZs[tIdx], self.transVecXYZs[tIdx],\
            extrViewMat, intrProjMat, rela_R, rela_txyz, rowScaleXZ, stage=1)
        if verbose == True:
            print("errorPixel: {:.2f}, errorAniScales: {:.2f}".format(errorPixel, errorAniScales))
        return errorPixel + errorAniScales 
        
    
    def maximization_stage1_step(self, photoType):
        # 优化 ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz
        ph = photoType.value
        # prepare input x0
        x0, pIdx = self.getGlobalParamsOfSingleView_as_x0(photoType,\
            self.ex_rxyz, self.ex_txyz, self.focLth, self.dpix, self.u0, self.v0, self.rela_rxyz, self.rela_txyz)
        # optimize
        resOptParams = scipy.optimize.minimize(fun=self.residualErrorOfGlobalParams, x0=x0, \
            args=(pIdx, photoType, False, False), method="COBYLA", options={"rhobeg":1.0,"maxiter":1000}) # without rowScaleXZ
        params = resOptParams.x
        # update
        self.ex_rxyz[ph], self.ex_txyz[ph], self.focLth[ph], self.dpix[ph], self.u0[ph], self.v0[ph], self.rela_rxyz, self.rela_txyz =\
            self.parseGlobalParamsOfSingleView(params, pIdx, photoType)


    def residualErrorOfGlobalParamsOf5Views(self, params, pIdx, verbose):
        # params = np.array([ex_ryz, ex_txyz, f, dpix, u0, v0, rowScaleXZ])
        ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz = self.parseGlobalParamsOf5Views(params, pIdx)
        rowScaleXZ = params[pIdx["rowScaleXZ"]:pIdx["rowScaleXZ"]+2]
        errors = []
        for phType in self.photoTypes:
            sub_x0, sub_pIdx = self.getGlobalParamsOfSingleView_as_x0(phType, ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz)
            sub_pIdx["rowScaleXZ"] = len(sub_x0)
            sub_x0 = np.hstack([sub_x0, rowScaleXZ])
            ph = phType.value
            sub_error = self.weightViewMaxiStage1[ph] * self.residualErrorOfGlobalParams(sub_x0, sub_pIdx, phType, withAniScales=True, verbose=verbose)
            errors.append(sub_error)
        if verbose==True:
            print("maximization stage1 step errors: [{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]".format(*errors))
        return np.sum(errors)
        
    
    def maximization_stage1_step_5Views(self, rhobeg=1.0, maxiter=1000, verbose=True):
        # 优化 ex_rxyz, ex_txyz 与 focLth, dpix, u0, v0, 以及 rela_rxyz, rela_txyz, rowScaleXZ
        x0, pIdx = self.getCurrentGlobalParamsOf5Views_as_x0()
        pIdx["rowScaleXZ"] = len(x0)
        x0 = np.hstack([x0, self.rowScaleXZ])

        resOptParams = scipy.optimize.minimize(fun=self.residualErrorOfGlobalParamsOf5Views, x0=x0, \
            args=(pIdx,False), method="COBYLA", options={"rhobeg":rhobeg,"maxiter":maxiter})
        params = resOptParams.x

        self.ex_rxyz, self.ex_txyz, self.focLth, self.dpix, self.u0, self.v0, self.rela_rxyz, self.rela_txyz =\
            self.parseGlobalParamsOf5Views(params, pIdx)
        self.rowScaleXZ = params[pIdx["rowScaleXZ"]:pIdx["rowScaleXZ"]+2]

        self.loss_maximization_step = self.residualErrorOfGlobalParamsOf5Views(params, pIdx, verbose) # For print purpose



    ###########################################
    ######### Maximization stage 2 ############
    ###########################################

    def anistropicRowScale2ScalesAndTransVecs(self):
        # 将各向异性牙列放缩在stage2之前转化为scale和transVecXYZs
        # self.meanScales = np.prod(self.rowScaleXZ)**(1/3) * np.ones_like(self.scales, np.float32)
        # self.meanTransVecXYZs[:,[0,2]] = self.X_Mu_centroids[:,[0,2]] * (self.rowScaleXZ - 1.)
        self.scales = np.prod(self.rowScaleXZ)**(1/3) * np.ones_like(self.scales, np.float32)
        self.transVecXYZs[:,[0,2]] = self.X_Mu_centroids[:,[0,2]] * (self.rowScaleXZ - 1.)
        self.rowScaleXZ = np.array([1.,1.], np.float32)


    def computeTeethPoseResidualError(self, scales, rotAngleXYZs, transVecXYZs, ph, tIdx):
        centeredPoseParams = np.hstack([(transVecXYZs-self.meanTransVecXYZs[tIdx]), (rotAngleXYZs-self.meanRotAngleXYZs[tIdx])]) # shape=(self.numTooth,7)
        centeredScales = scales - self.meanScales[tIdx]
        errorTeethPose = np.sum(np.matmul(np.matmul(centeredPoseParams[:,None,:], self.invCovMats[tIdx,:,:]), centeredPoseParams[:,:,None]))
        errorScales = centeredScales @ self.invCovMatOfScale[tIdx,tIdx[:,None]] @ centeredScales
        return self.weightTeethPose * (errorTeethPose + errorScales)

    
    def residualErrorOfTeethPose(self, params, pIdx, photoType, verbose, step=1):
        ph = photoType.value
        tIdx = self.visIdx[ph]

        ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz = self.parseGlobalParamsOfSingleView(params, pIdx, photoType)
        extrViewMat = self.updateExtrinsicViewMatrix(ex_rxyz, ex_txyz)
        intrProjMat = self.updateIntrinsicProjectionMatrix(focLth, dpix, u0, v0)
        rela_R = self.updateRelaRotMat(rela_rxyz)
        transVecXYZs, rotAngleXYZs, scales = self.parseTeethPoseParams(params, pIdx, tIdx, step)

        errorPixel = self.computePixelResidualError(photoType, self.featureVec, scales, rotAngleXYZs, transVecXYZs, \
            extrViewMat, intrProjMat, rela_R, rela_txyz, stage=2) # negative log likelihood of pixel distance distribution
        errorTeethPoseParam = self.computeTeethPoseResidualError(scales, rotAngleXYZs, transVecXYZs, ph, tIdx)
        if verbose == True:
            print("{}, total error: {:.2f}, errorPixel:{:.2f}, errorTeethPoseParam:{:.2f}".format(str(photoType), errorPixel+errorTeethPoseParam, errorPixel, errorTeethPoseParam))
        return errorPixel + errorTeethPoseParam


    def maximization_stage2_step(self, photoType, step=1, verbose=True):
        ph = photoType.value
        tIdx = self.visIdx[ph]
        numT = len(tIdx)

        x0, pIdx = self.getGlobalParamsOfSingleView_as_x0(photoType,\
            self.ex_rxyz, self.ex_txyz, self.focLth, self.dpix, self.u0, self.v0, self.rela_rxyz, self.rela_txyz)

        rhobeg = 1.0
        maxiter = 1000
        alpha = 0.1

        if step == 1:
            pIdx["tXYZs"] = len(x0)
            x0 = np.hstack([x0, self.transVecXYZs[tIdx].flatten()])
            rhobeg = alpha * self.transVecStd
        elif step == 2:
            pIdx["rXYZs"] = len(x0)
            x0 = np.hstack([x0, self.rotAngleXYZs[tIdx].flatten()])
            rhobeg = alpha * self.rotAngleStd
        elif step == 3:
            pIdx["scales"] = len(x0)
            x0 = np.hstack([x0, self.scales[tIdx]])
            rhobeg = alpha * self.scaleStd
        elif step == 4:
            pIdx.update({"tXYZs":len(x0), "rXYZs":len(x0)+numT*3, "scales":len(x0)+numT*6})
            x0 = np.hstack([x0, (self.transVecXYZs[tIdx]/self.transVecStd).flatten(), (self.rotAngleXYZs[tIdx]/self.rotAngleStd).flatten(), (self.scales[tIdx]-1.)/self.scaleStd])
        
        optRes = scipy.optimize.minimize(fun=self.residualErrorOfTeethPose, x0=x0, args=(pIdx, photoType, False, step), \
            method="COBYLA", options={"rhobeg":rhobeg,"maxiter":maxiter})
        params = optRes.x

        # update
        self.ex_rxyz[ph], self.ex_txyz[ph], self.focLth[ph], self.dpix[ph], self.u0[ph], self.v0[ph], self.rela_rxyz, self.rela_txyz =\
            self.parseGlobalParamsOfSingleView(params, pIdx, photoType)
        self.transVecXYZs[tIdx], self.rotAngleXYZs[tIdx], self.scales[tIdx] = self.parseTeethPoseParams(self, params, pIdx, tIdx, step)

        self.residualErrorOfTeethPose(params, pIdx, photoType, verbose, step) # For print
    
    
    def residualErrorOfTeethPoseOf5Views(self, params, pIdx, verbose, step=1):
        ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz = self.parseGlobalParamsOf5Views(params, pIdx)
        tIdx = [i for i in range(self.numTooth)]
        transVecXYZs, rotAngleXYZs, scales = self.parseTeethPoseParams(params, pIdx, tIdx, step)

        errors = []
        for phType in self.photoTypes:
            ph = phType.value
            tIdx = self.visIdx[ph]
            numT = len(tIdx)
            sub_x0, sub_pIdx = self.getGlobalParamsOfSingleView_as_x0(phType, ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz)
            if step == 1:
                sub_pIdx["tXYZs"] = len(sub_x0)
                sub_x0 = np.hstack([sub_x0, transVecXYZs[tIdx].flatten()])
            elif step == 2:
                sub_pIdx["rXYZs"] = len(sub_x0)
                sub_x0 = np.hstack([sub_x0, rotAngleXYZs[tIdx].flatten()])
            elif step == 3:
                sub_pIdx["scales"] = len(sub_x0)
                sub_x0 = np.hstack([sub_x0, scales[tIdx]])
            elif step == 4:
                sub_pIdx.update({"tXYZs":len(sub_x0), "rXYZs":len(sub_x0)+numT*3, "scales":len(sub_x0)+numT*6})
                sub_x0 = np.hstack([sub_x0, (transVecXYZs[tIdx]/self.transVecStd).flatten(), \
                    (rotAngleXYZs[tIdx]/self.rotAngleStd).flatten(), (scales[tIdx]-1.)/self.scaleStd])

            sub_error = self.weightViewMaxiStage2[ph] * self.residualErrorOfTeethPose(sub_x0, sub_pIdx, phType, verbose, step)
            errors.append(sub_error)
        if verbose==True:
            print("maximization step errors: [{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]".format(*errors))
        return np.sum(errors)


    def maximization_stage2_step_5Views(self, tol=1e-6, rhobeg=0.1, maxiter=1000, step=1, verbose=True):
        x0, pIdx = self.getCurrentGlobalParamsOf5Views_as_x0()
        if step == 1:
            pIdx["tXYZs"] = len(x0)
            x0 = np.hstack([x0, self.transVecXYZs.flatten()])
            rhobeg = rhobeg * self.transVecStd
        elif step == 2:
            pIdx["rXYZs"] = len(x0)
            x0 = np.hstack([x0, self.rotAngleXYZs.flatten()])
            rhobeg = rhobeg * self.rotAngleStd
        elif step == 3:
            pIdx["scales"] = len(x0)
            x0 = np.hstack([x0, self.scales])
            rhobeg = rhobeg * self.scaleStd
        elif step == 4:
            pIdx.update({"tXYZs":len(x0), "rXYZs":len(x0)+self.numTooth*3, "scales":len(x0)+self.numTooth*6})
            x0 = np.hstack([x0, (self.transVecXYZs/self.transVecStd).flatten(), (self.rotAngleXYZs/self.rotAngleStd).flatten(), (self.scales-1.)/self.scaleStd])
        
        optRes = scipy.optimize.minimize(fun=self.residualErrorOfTeethPoseOf5Views, x0=x0, args=(pIdx, False, step), \
            method="COBYLA", tol=tol, options={"rhobeg":rhobeg,"maxiter":maxiter})
        params = optRes.x

        self.ex_rxyz, self.ex_txyz, self.focLth, self.dpix, self.u0, self.v0, self.rela_rxyz, self.rela_txyz =\
            self.parseGlobalParamsOf5Views(params, pIdx)
        tIdx = [i for i in range(self.numTooth)]
        self.transVecXYZs, self.rotAngleXYZs, self.scales = self.parseTeethPoseParams(params, pIdx, tIdx, step)

        self.loss_maximization_step = self.residualErrorOfTeethPoseOf5Views(params, pIdx, verbose, step)


    ###########################################
    ######### Maximization stage 3 ############
    ###########################################
    
    def residualErrorOfFeatureVec(self, featureVecParam, verbose):
        featureVec = featureVecParam.reshape((self.numTooth,1,self.numPC))
        errors = []
        for phType in self.photoTypes:
            ph = phType.value
            tIdx = self.visIdx[ph]
            errorPixel = self.computePixelResidualError(phType, featureVec[tIdx], self.scales[tIdx], self.rotAngleXYZs[tIdx], self.transVecXYZs[tIdx], self.extrViewMat[ph], self.intrProjMat[ph],\
                self.rela_R, self.rela_txyz, rowScaleXZ=np.ones((2,),np.float32), stage=3)
            errors.append(errorPixel)
        errorFeatureVec = self.weightFeatureVec * np.sum(featureVec**2)
        if verbose == True:
            print("errorPixel:{:.2f}, errorFeatureVec:{:.2f}".format(np.sum(errors), errorFeatureVec))
        return errorPixel + errorFeatureVec
    
    def updateCorreSigmaTSegs(self, photoType):
        ph = photoType.value
        tIdx = self.visIdx[ph]
        SigmaT_segs = []
        for sigmaT,vis_hull_vids in zip(self.SigmaT[tIdx], self.vis_hull_vertex_indices[ph]): # self.SigmaT.shape=(numTooth,numPC,numPoint*3)
            sigmaTseg = sigmaT.reshape(self.numPC, self.numPoint, 3)[:,vis_hull_vids,:]
            SigmaT_segs.append(sigmaTseg.reshape(self.numPC, 3*len(vis_hull_vids)))
        return SigmaT_segs
            
    def maximization_stage3_step(self, verbose):
        for phType in self.photoTypes:
            self.SigmaT_segs[phType.value] = self.updateCorreSigmaTSegs(phType)
        resOptFeatureVec = scipy.optimize.minimize(fun=self.residualErrorOfFeatureVec, x0=self.featureVec.flatten(), args=(False,), method="COBYLA", options={"rhobeg":1.0,"maxiter":1000})
        self.featureVec = np.reshape(resOptFeatureVec.x, self.featureVec.shape)
        if verbose == True:
            self.residualErrorOfFeatureVec(resOptFeatureVec.x, True)


    ###########################################
    ######### Visualization ###################
    ###########################################

    
    def showEdgeMaskPredictionWithGroundTruth(self, photoType, canvasShape=None, dilate=True):
        # red: prediction, white: ground truth
        ph = photoType.value
        if not bool(canvasShape):
            canvasShape = self.edgeMaskU[ph].shape
        canvas = np.zeros((*canvasShape,3), dtype=np.float32)
        h, w = self.edgeMaskU[ph].shape
        canvas[:h,:w,:] = np.stack([self.edgeMaskU[ph],self.edgeMaskL[ph],np.zeros((h,w))],axis=-1) # white: ground truth
        
        edgePred = np.zeros(canvasShape, dtype=np.float32)
        P_pred = None
        if photoType == PHOTO.UPPER:
            P_pred = self.P_pred_U[ph]
        elif photoType == PHOTO.LOWER:
            P_pred = self.P_pred_L[ph]
        else:
            P_pred = np.vstack([self.P_pred_U[ph],self.P_pred_L[ph]])
        edgePred[P_pred[:,1], P_pred[:,0]] = 1. # red: edge prediction
        if dilate == True:
            edgePred = skimage.morphology.binary_dilation(edgePred, skimage.morphology.disk(2)) # dilation edge prediction for visualization
        canvas[:,:,2] = np.max(np.stack([edgePred,canvas[:,:,2]]), axis=0)
        
        plt.figure(figsize = (10,10))
        plt.imshow(canvas)