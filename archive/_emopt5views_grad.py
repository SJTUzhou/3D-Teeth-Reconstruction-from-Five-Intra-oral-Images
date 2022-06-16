from pickle import NONE
import numpy as np
import pandas as pd
import open3d as o3d
from matplotlib import pyplot as plt
from quaternion import as_quat_array
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
    # 使用梯度下降类的算法进行优化
    def __init__(self, edgeMasks, photoTypes, visMasks, Mask, Mu, SqrtEigVals, Sigma, PoseCovMats, ScaleCovMat, transVecStd, rotAngleStd) -> None:
        self.photoTypes = sorted(photoTypes, key=lambda x:x.value)
        assert self.photoTypes == [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
        
        self.edgeMask = [None] * 5 # order as the Enum value in PHOTO
        self.visIdx = [None] * 5 # 每张照片中出现的牙齿轮廓的牙齿的Mask
        for phType, eMask, visMask in zip(photoTypes, edgeMasks, visMasks):
            assert eMask.ndim == 2, "edgeMask should be grayscale" # 单通道Mask图片
            self.edgeMask[phType.value] = eMask # binary 2d-array
            self.visIdx[phType.value] = np.argwhere(visMask[Mask]>0).flatten()
        
        self.P_true = [np.argwhere(v>0)[:,::-1] for v in self.edgeMask] # real edge point pos in image coord (u_x,v_y), 2d-array, shape=(?,2)
        self.M = [len(v) for v in self.P_true] # 真实Mask中边缘像素点的数量
        self.P_true_normals = [self.__initEdgeMaskNormals(v) for v in self.P_true]
        
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
        
        self.aniScales = np.ones((3,), dtype=np.float32) # 各向异性牙列放缩, 仅用于maximization stage1, 在stage2之前转化为scale和transVecXYZs

        # init extrinsic param of camera
        ex_rxyz = np.vstack([self.__initExtrinsicRotAngles(k) for k in self.photoTypes]) # shape=(5,3) # init rot angles around x-y-z axis based on photoType
        self.ex_q = self.euler2q(ex_rxyz)

        obj_dist_estimates = [45.,45.,35.,35.,45.]
        self.ex_txyz = np.vstack([self.__initExtrinsicTransVec(k, obj_dist_estimate=obj_d_est)\
             for k,obj_d_est in zip(self.photoTypes,obj_dist_estimates)]) # shape=(5,3) # init trans vector
        self.rela_q = np.array([0.,0.,0.,1.],dtype=np.float32) #下牙列相对于上牙列的旋转
        self.rela_R = self.updateRelaRotMat(self.rela_q)
        self.rela_txyz = np.array([0.,-5.,0.],dtype=np.float32) #下牙列相对于上牙列的位移

        self.ex_q_lr = 1.
        self.ex_txyz_lr = 1.
        self.fx_lr = 1.
        self.uv_lr = 1.
        self.rela_q_lr = 1.
        self.rela_txyz_lr = 1.
        
        # init intrinsic param of camera
        focLth =  np.array([50.0,50.0,35.0,35.0,35.0],dtype=np.float32)
        dpix = 0.06
        self.fx = focLth / dpix
        self.u0 = np.array([v.shape[1]//2 for v in self.edgeMask], dtype=np.uint32) # img.width//2
        self.v0 = np.array([v.shape[0]//2 for v in self.edgeMask], dtype=np.uint32) # img.height//2
        
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
        self.P_pred = [None] * 5 # edgeMask prediction 2d-array, shape=(?,2)
        self.P_pred_normals = [None] * 5
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

    @staticmethod
    def computePointNormals(X):
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
    @staticmethod
    def __initEdgeMaskNormals(vertices_xy):
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
    
    @staticmethod
    def __initExtrinsicRotAngles(photoType):
        if photoType == PHOTO.UPPER:
            return np.array([-0.3*np.pi, np.pi, np.pi], dtype=np.float32) # upper
        elif photoType == PHOTO.LOWER:
            return np.array([0.3*np.pi, np.pi, np.pi], dtype=np.float32)  # lower
        elif photoType == PHOTO.LEFT:
            return np.array([0.0, 0.7*np.pi, np.pi], dtype=np.float32) # left
        elif photoType == PHOTO.RIGHT:
            return np.array([0.0, -0.7*np.pi, np.pi], dtype=np.float32)  # right
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
            return np.array([0., -2., -self.X_Mu[...,2].min()+obj_dist_estimate], dtype=np.float32)  # frontal
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
    @staticmethod
    def computeTeethRotMats(rotAngleXYZs):
        rotMats = RR.from_euler("xyz", rotAngleXYZs).as_matrix()
        rotMats = np.transpose(rotMats, (0,2,1)) # 变为右乘
        return rotMats
    
    def updateTransformedPointPos(self, X_deformed, scales, rotMats, transVecXYZs, tIdx):
        # X_trans = scales_inv * (X_deformed + transVecXYZs_inv) @ rotMats_inv
        # in CPD: X_aligned_deformed = scales_cpd * X_trans @ rotMats_cpd + transVecXYZs_cpd
        return np.multiply(scales[:,None,None], np.matmul(X_deformed-self.X_Mu_centroids[tIdx,None,:], rotMats)) +\
                transVecXYZs[:,None,:] + self.X_Mu_centroids[tIdx,None,:]
    
    @staticmethod
    def updateTransformedPointNormals(X_deformed_normals, rotMats):
        # 法向量只需要计算旋转即可
        return np.matmul(X_deformed_normals, rotMats)


    ####################################################################
    # Relative Pose of Lower Tooth Row with respect to Upper Tooth Row #
    ####################################################################
    @staticmethod
    def updateRelaRotMat(rela_q):
        return RR.from_quat(rela_q).as_matrix().T
    @staticmethod
    def updateLowerPointPosByRelaPose(X_lower, rela_R, rela_txyz):
        return np.matmul(X_lower, rela_R) + rela_txyz
    @staticmethod
    def updateLowerPointNormalsByRelaPose(X_lower_normals, rela_R):
        return np.matmul(X_lower_normals, rela_R)


    ###############################
    # world coord -> camera coord #
    ###############################
    @staticmethod
    def updateExtrinsicViewMatrix(ex_q, ex_txyz): # world coord to camera coord
        # 先进行x轴旋转，再y轴，再z轴；取转置表示右乘旋转矩阵，再平移
        R = RR.from_quat(ex_q).as_matrix().T
        return np.vstack([R, ex_txyz]) # Matrix 4*3
    @staticmethod
    def updatePointPosInCameraCoord(X_world, extrViewMat):
        # get 3D point cloud in camera coord, return array shape (n,3) or (batch,n,3)
        X_homo = np.concatenate([X_world, np.ones((*X_world.shape[:-1],1))], axis=-1)
        return np.matmul(X_homo, extrViewMat)
    @staticmethod
    def updatePointNormalsInCameraCoord(X_world_normals, extrViewRotMat):
        return np.matmul(X_world_normals, extrViewRotMat)


    
    
    ##############################
    # camera coord ->image coord #
    ##############################
    @staticmethod
    def updateIntrinsicProjectionMatrix(fx, u0, v0): # camera cood to image coord
        return np.array([[fx, 0., 0.], [0., fx, 0.], [u0, v0, 1.]])
    @staticmethod
    def updatePointPosInImageCoord(X_camera, intrProjMat):
        # get 3D point cloud in image coord, return array shape (n,2) or (batch,n,2)
        assert (X_camera[...,2]>0).all() # Z-value of points should be positive
        X_image = np.matmul((X_camera/X_camera[...,[2]]), intrProjMat)
        X_uv = X_image[...,:2]
        return np.around(X_uv).astype(np.int32)
    @staticmethod
    def updatePointNormalsInImageCoord(X_camera_normals):
        X_cam_normals_xy = X_camera_normals[...,:2] # 取相机坐标系下normals的x,y坐标即为图片坐标系中的normals
        return X_cam_normals_xy / np.linalg.norm(X_cam_normals_xy, axis=-1, keepdims=True)


    ##################################################
    # Extract contour pixels in projected pointcloud #
    ##################################################
    @staticmethod
    def __getUniquePixels(X_uv_int):
        # merge points at the same position in image coord
        # X_uv_int: array shape (n,2)dtype np.int32
        # pixels: array (m,2), each element represents (u_x, v_y)
        pixels, unique_indices = np.unique(X_uv_int,axis=0,return_index=True)
        return pixels, unique_indices
    @staticmethod
    def __getConcaveHullEdgeVertexIndices(coords, alpha):  # coords is a 2D numpy array (u_x,v_y)
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
    @staticmethod
    def __constructConcaveHull(coords, edge_indices): # coords is a 2D numpy array (u_x,v_y)
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
        self.X_trans[tIdx] = self.aniScales * self.X_trans[tIdx] # self.aniScales = [1.,1.,1.] after maximization stage 1
        self.X_trans_normals[tIdx] = self.updateTransformedPointNormals(self.X_deformed_normals[tIdx], self.RotMats[tIdx])

    def updateEdgePrediction(self, photoType):
        # 根据拍摄角度，决定使用上牙列或下牙列点云
        ph = photoType.value
        tIdx = self.visIdx[ph]
        X_trans = self.X_trans[tIdx] # upper
        X_trans_normals = self.X_trans_normals[tIdx]

        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            self.rela_R = self.updateRelaRotMat(self.rela_q)
            X_trans = np.concatenate([X_trans[:ul_sp], \
                self.updateLowerPointPosByRelaPose(X_trans[ul_sp:], self.rela_R, self.rela_txyz)], axis=0) # left, right, frontal
            X_trans_normals = np.concatenate([X_trans_normals[:ul_sp], \
                self.updateLowerPointNormalsByRelaPose(X_trans_normals[ul_sp:], self.rela_R)], axis=0)
        
        self.extrViewMat[ph] = self.updateExtrinsicViewMatrix(self.ex_q[ph], self.ex_txyz[ph]) # homo world coord (xw,yw,zw,1) to camera coord (xc,yc,zc): 4*3 right-multiplying matrix
        self.X_camera[ph] = self.updatePointPosInCameraCoord(X_trans, self.extrViewMat[ph]) # compute X in camera coord based on X in world coord, ndarray, shape=(numTooth,1500,3)
        self.X_camera_normals[ph] = self.updatePointNormalsInCameraCoord(X_trans_normals, self.extrViewMat[ph,:3,:])
        
        self.intrProjMat[ph] = self.updateIntrinsicProjectionMatrix(self.fx[ph], self.u0[ph], self.v0[ph]) # camera coord (xc,yc,zc) to image coord (u,v,zc): 3*3 right-multiplying matrix
        self.X_uv[ph] = self.updatePointPosInImageCoord(self.X_camera[ph], self.intrProjMat[ph]) # compute X in image coord based on X_camera in camera coord, ndarray, shape=(numTooth,1500,2)
        self.X_uv_normals[ph] = self.updatePointNormalsInImageCoord(self.X_camera_normals[ph])
        
        self.vis_hull_vertices[ph], self.vis_hull_vertex_indices[ph] = self.extractVisibleEdgePointsByAvgDepth(photoType) # visible points in image coord, and corre idx in X
        self.P_pred[ph] = np.vstack(self.vis_hull_vertices[ph]) # edgeMask prediction 2d-array, shape=(?,2)
        self.P_pred_normals[ph] = np.vstack([x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_uv_normals[ph], self.vis_hull_vertex_indices[ph])])  # edgeMask normals prediction 2d-array, shape=(?,2)
        
        self.X_Mu_pred[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_Mu[tIdx], self.vis_hull_vertex_indices[ph])] # points in world coord corre to edgeMask prediction
        self.X_Mu_pred_normals[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_Mu_normals[tIdx], self.vis_hull_vertex_indices[ph])]
        self.X_deformed_pred[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_deformed[tIdx], self.vis_hull_vertex_indices[ph])] 
        self.X_deformed_pred_normals[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_deformed_normals[tIdx], self.vis_hull_vertex_indices[ph])] 
        
    
    
    ###########################################
    ######### Update & Expectation Step #######
    ###########################################
    
    def expectation_step(self, photoType, verbose=True):
        # 根据新的edgePredcition计算对应点对关系
        ph = photoType.value
        self.updateAlignedPointCloudInWorldCoord(tIdx=self.visIdx[ph])
        self.updateEdgePrediction(photoType)

        point_loss_mat = distance_matrix(self.P_true[ph], self.P_pred[ph], p=2, threshold=int(1e8))**2
        normal_loss_mat = - (self.P_true_normals[ph] @ self.P_pred_normals[ph].T)**2 / self.varAngle
        loss_mat = point_loss_mat * np.exp(normal_loss_mat)
        self.corre_pred_idx[ph] = np.argmin(loss_mat, axis=1)
        losses = loss_mat[np.arange(self.M[ph]), self.corre_pred_idx[ph]]
        self.loss_expectation_step[ph] = np.sum(losses)
        if verbose==True:
            print("{} - unique pred points: {} - E-step loss: {:.2f}".format(str(photoType), len(np.unique(self.corre_pred_idx[ph])), self.loss_expectation_step[ph]))

    def expectation_step_5Views(self, verbose=True):
        # 根据新的edgePredcition计算对应点对关系,对5张图同时进行
        tIdx = [i for i in range(self.numTooth)]
        self.updateAlignedPointCloudInWorldCoord(tIdx)
        for photoType in self.photoTypes:
            ph = photoType.value
            self.updateEdgePrediction(photoType)
            point_loss_mat = distance_matrix(self.P_true[ph], self.P_pred[ph], p=2, threshold=int(1e8))**2
            normal_loss_mat = - (self.P_true_normals[ph] @ self.P_pred_normals[ph].T)**2 / self.varAngle
            loss_mat = point_loss_mat * np.exp(normal_loss_mat)
            self.corre_pred_idx[ph] = np.argmin(loss_mat, axis=1)
            losses = loss_mat[np.arange(self.M[ph]), self.corre_pred_idx[ph]]
            self.loss_expectation_step[ph] = np.sum(losses)
            if verbose==True:
                print("{} - unique pred points: {} - E-step loss: {:.2f}".format(str(photoType), len(np.unique(self.corre_pred_idx[ph])), self.loss_expectation_step[ph]))

    ###########################################
    ######### Maximization Step ###############
    ###########################################
    def computePixelResidualError(self, photoType, featureVec, scales, rotAngleXYZs, transVecXYZs, extrViewMat, intrProjMat,\
        rela_R, rela_txyz, aniScales=np.ones((3,),np.float32), stage=1):
        # self.X_?_pred: List of array of points in Mu teeth shape, [ndarray1, ndarray2, ...]
        # self.corre_pred_idx: corre indices after vertically stacking the transformed self.X_?_pred
        
        ph = photoType.value
        tIdx = self.visIdx[ph]
        X_deformed_pred = self.X_Mu_pred[ph]
        _X_trans_pred = self.X_Mu_pred[ph] # _ 表示未考虑上下牙列位置关系
        X_deformed_pred_normals = self.X_Mu_pred_normals[ph]
        _X_trans_pred_normals = self.X_Mu_pred_normals[ph]


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
            _X_trans_pred = [s*np.matmul(x-tc,R)+t+tc for x,s,R,t,tc in zip(X_deformed_pred, scales, rotMats, transVecXYZs, self.X_Mu_centroids[tIdx])] # 轮廓点对应原始点云按牙齿分别进行缩放刚性变换
            _X_trans_pred_normals = [np.matmul(xn,R) for xn,R in zip(X_deformed_pred_normals, rotMats)]
            
        
        # 需要考虑上下牙列位置关系，对下牙列的点进行相对旋转和平移
        X_trans_pred = _X_trans_pred
        X_trans_pred_normals = _X_trans_pred_normals
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            X_trans_pred = _X_trans_pred[:ul_sp] + [x@rela_R+rela_txyz for x in _X_trans_pred[ul_sp:]]
            X_trans_pred_normals = _X_trans_pred_normals[:ul_sp] + [xn@rela_R for xn in _X_trans_pred_normals[ul_sp:]]
        
        _X_corre_pred = np.vstack(X_trans_pred)[self.corre_pred_idx[ph]]  # _ 表示未考虑牙列各向异性放缩 
        X_corre_pred = _X_corre_pred       
        if stage == 1: # 在优化相机参数同时，优化牙列的Anistropic scales
            X_corre_pred = aniScales * _X_corre_pred
        X_corre_pred_normals = np.vstack(X_trans_pred_normals)[self.corre_pred_idx[ph]]

        X_cam_corre_pred = self.updatePointPosInCameraCoord(X_corre_pred, extrViewMat) #相机坐标系下对应点坐标
        X_cam_corre_pred_normals = self.updatePointNormalsInCameraCoord(X_corre_pred_normals, extrViewMat[:3,:]) # extrViewMat.shape = (4,3)
        
        P_corre_pred = self.updatePointPosInImageCoord(X_cam_corre_pred, intrProjMat)
        P_corre_pred_normals = self.updatePointNormalsInImageCoord(X_cam_corre_pred_normals) # \hat{ni}
        
        ci_hatci = self.P_true[ph] - P_corre_pred # ci - \hat{ci}
        resPointError = np.sum(np.linalg.norm(ci_hatci, axis=1)**2) / self.varPoint
        ci_hatci_dot_hatni = np.sum(ci_hatci*P_corre_pred_normals, axis=1)
        resPlaneError = np.sum(ci_hatci_dot_hatni**2) / self.varPlane
        loss = (resPointError + resPlaneError) / self.M[ph]

        # 计算梯度
        hatni = P_corre_pred_normals
        par_loss_par_hatci = 1./ (0.5*self.M[ph]) * np.matmul(-ci_hatci[:,None,:], \
            (1./self.varPoint * np.identity(2,np.float32) + 1./self.varPlane * np.matmul(hatni[:,:,None],hatni[:,None,:]))) #(self.M[ph], 1, 2)
        par_loss_par_hatni = 1./(0.5*self.varPlane*self.M[ph]) * ci_hatci_dot_hatni[:,None,None] * ci_hatci.reshape(self.M[ph],1,2)  #(self.M[ph], 1, 2)

        g = X_cam_corre_pred # 3d-point after global transformation
        gn = X_cam_corre_pred_normals
        gz = g[:,[2]]
        gxgy_gz = g[:,:2] / gz # (self.M[ph],2)
        par_hatci_par_fx = gxgy_gz[...,None] # (self.M[ph],2,1)
        par_hatci_par_u0 = np.array([[1.],[0.]],np.float32)
        par_hatci_par_v0 = np.array([[0.],[1.]],np.float32)

        # 对于相机内参的梯度
        grad_fx = np.sum(np.matmul(par_loss_par_hatci, par_hatci_par_fx))
        grad_u0 = np.sum(np.matmul(par_loss_par_hatci, par_hatci_par_u0))
        grad_v0 = np.sum(np.matmul(par_loss_par_hatci, par_hatci_par_v0))

        fx = intrProjMat[0,0]
        par_hatci_par_g = fx/gz[:,:,None] * np.concatenate([np.tile(np.identity(2,np.float32),(self.M[ph],1,1)), -gxgy_gz[...,None]],axis=-1) # (self.M[ph],2,3)
        par_loss_par_g = np.matmul(par_loss_par_hatci, par_hatci_par_g) # (self.M[ph],1,3)

        # 对于ex_txyz的梯度
        # par_g_par_ext = np.identity(3,np.float32) 
        grad_ext = np.sum(par_loss_par_g, axis=0) # (1,3)

        par_hatni_par_gn = self.jacobs_hatni_wrt_gn(gn) # (self.M[ph],2,3)
        par_loss_par_gn = np.matmul(par_loss_par_hatni, par_hatni_par_gn)  # (self.M[ph],1,3)

        R_global = extrViewMat[:3,:].T
        exq = self.R2q(R_global)

        p = _X_corre_pred # before aniScale
        sp = X_corre_pred # after aniScale
        pn = X_corre_pred_normals
        _par_g_par_exq = np.stack([self.jacob_q_wrt_point(exq, a) for a in sp], axis=0)
        _par_gn_par_exq = np.stack([self.jacob_q_wrt_point(exq, an) for an in pn], axis=0)
        _jacob_norm_exq = self.jacob_normlize_q(exq)
        par_g_par_exq = np.matmul(_par_g_par_exq, _jacob_norm_exq) # (self.M[ph],3,4)
        par_gn_par_exq = np.matmul(_par_gn_par_exq, _jacob_norm_exq) # (self.M[ph],3,4)
        # 对于ex_rxyz转换的单位四元数（实数部非负）的梯度
        grad_exq = np.sum(np.matmul(par_loss_par_g, par_g_par_exq), axis=0) + np.sum(np.matmul(par_loss_par_gn, par_gn_par_exq), axis=0) # (1,4) 

        # 对于牙列各向异性放缩的梯度
        grad_aniScale = np.zeros((1,3),np.float32)
        # if stage == 1:
        #     par_g_par_aniScale = np.matmul(R_global, np.stack([np.diag(a) for a in p],axis=0)) # (self.M[ph],3,3)
        #     grad_aniScale = np.sum(np.matmul(par_loss_par_g, par_g_par_aniScale), axis=0) # (1,3)

        # 计算牙列相对位置关系参数的梯度
        relaq = self.R2q(rela_R.T)
        grad_relaq = np.zeros((1,4),np.float32)
        grad_relat = np.zeros((1,3),np.float32)
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            ks = len(np.vstack(_X_trans_pred[:ul_sp]))
            idx_l = self.corre_pred_idx[ph] >= ks
            corre_pred_idx_l = self.corre_pred_idx[ph][idx_l] - ks
            
            _pl = np.vstack(_X_trans_pred[ul_sp:])[corre_pred_idx_l]   # 下牙列经过相对旋转位移前的点坐标
            _pln = np.vstack(_X_trans_pred_normals[ul_sp:])[corre_pred_idx_l]
            par_loss_par_gl = par_loss_par_g[idx_l]
            par_loss_par_gln = par_loss_par_gn[idx_l]
            _jacob_norm_relaq = self.jacob_normlize_q(relaq)
            _par_gl_par_relaq = np.matmul(R_global, np.stack([self.jacob_q_wrt_point(relaq, a) for a in _pl], axis=0))
            _par_gln_par_relaq = np.matmul(R_global, np.stack([self.jacob_q_wrt_point(relaq, an) for an in _pln], axis=0))
            par_gl_par_relaq = np.matmul(_par_gl_par_relaq, _jacob_norm_relaq) # (self.ML[ph],3,4)
            par_gln_par_relaq = np.matmul(_par_gln_par_relaq, _jacob_norm_relaq) # (self.ML[ph],3,4)
            par_gl_par_relat = R_global
            grad_relaq = np.sum(np.matmul(par_loss_par_gl, par_gl_par_relaq), axis=0) + np.sum(np.matmul(par_loss_par_gln, par_gln_par_relaq), axis=0) # (1,4)
            grad_relat = np.sum(np.matmul(par_loss_par_gl, par_gl_par_relat), axis=0) # (1,3)
        
        grad = np.hstack([np.squeeze(grad_exq), np.squeeze(grad_ext), grad_fx, grad_u0, grad_v0, np.squeeze(grad_relaq), np.squeeze(grad_relat), np.squeeze(grad_aniScale)])
        
        return loss, grad

    @staticmethod
    def R2q(rotMat):
        q = RR.from_matrix(rotMat).as_quat() # scalar last
        if q[-1] < 0:
            q = -q # 确定4元数实部大于0
        return q

    @staticmethod
    def euler2q(eulerXYZ):
        q = RR.from_euler("xyz", eulerXYZ).as_quat() # scalar last, may be 2d array
        if q.ndim == 1 and q[-1] < 0:
            q = -q # 确定4元数实部大于0
        elif q.ndim == 2:
            mask = q[:,-1] < 0
            q[mask] = -q[mask]
        else:
            assert False
        return q

    @staticmethod
    def jacobs_hatni_wrt_gn(vec_gn):
        # vec_gn.shape = (m, 3), a list of point normals
        m = len(vec_gn)
        vec_gnx = vec_gn[:,0]
        vec_gny = vec_gn[:,1]
        vec_0 = np.zeros_like(vec_gnx, np.float32)
        vec_gnx_gny = vec_gnx * vec_gny
        vec_norm_gnxy = np.linalg.norm(vec_gn[:,:2], axis=1, keepdims=True)
        _jacob = np.stack([vec_gny**2, -vec_gnx_gny, vec_0, vec_gnx**2, -vec_gnx_gny, vec_0], axis=-1).reshape(m, 2, 3)
        return 1./(vec_norm_gnxy[:,:,None]**3) * _jacob
    
    @staticmethod
    def jacob_normlize_q(q): # "A tutorial on SE(3)": Equ. (1.7)
        assert len(q) == 4, "Not a quaternion" # scalar last
        q_norm = np.linalg.norm(q)
        _jacob = q_norm**2 * np.identity(4,np.float32) - q[:,None] * q
        return 1./(q_norm**3) * _jacob
    
    @staticmethod
    def jacob_q_wrt_point(q, a): # "A tutorial on SE(3)": Equ. (3.9) 前半部分
        # 一个点a经过四元数q旋转后变为a'，a'对q的jacobian matrix # scalar last
        assert len(a) == 3, "Not a 3d point"
        assert len(q) == 4, "Not a quaternion"
        ax, ay, az = a
        qx, qy, qz, qr = q
        _jacob = 2. * np.array([[qy*ay+qz*az, -2.*qy*ax+qx*ay+qr*az, -2.*qz*ax-qr*ay+qx*az, -qz*ay+qy*az],\
                                [qy*ax-2.*qx*ay-qr*az, qx*ax+qz*az, qr*ax-2.*qz*ay+qy*az, qz*ax-qx*az],\
                                [qz*ax+qr*ay-2.*qx*az, -qr*ax+qz*ay-2.*qy*az, qx*ax+qy*ay, -qy*ax+qx*ay]], dtype=np.float32)
        return _jacob

    def parseGlobalParamsOfSingleView(self, params, pIdx, photoType):
        ex_q = params[pIdx["ex_q"]:pIdx["ex_q"]+4]
        ex_txyz = self.ex_txyz_lr * params[pIdx["ex_txyz"]:pIdx["ex_txyz"]+3]
        fx = self.fx_lr * params[pIdx["fx"]]
        u0, v0 = self.uv_lr * params[pIdx["u0v0"]:pIdx["u0v0"]+2]
        rela_q = self.rela_q
        rela_txyz = self.rela_txyz
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            rela_q = params[pIdx["rela_q"]:pIdx["rela_q"]+4]
            rela_txyz = self.rela_txyz_lr * params[pIdx["rela_txyz"]:pIdx["rela_txyz"]+3]
        return ex_q, ex_txyz, fx, u0, v0, rela_q, rela_txyz

    def parseGlobalParamsOf5Views(self, params, pIdx):
        ex_q = params[pIdx["ex_q"]:pIdx["ex_q"]+20].reshape(5,4)
        ex_txyz = self.ex_txyz_lr * params[pIdx["ex_txyz"]:pIdx["ex_txyz"]+15].reshape(5,3)
        fx = self.fx_lr * params[pIdx["fx"]:pIdx["fx"]+5]
        u0 = self.uv_lr * params[pIdx["u0"]:pIdx["u0"]+5]
        v0 = self.uv_lr * params[pIdx["v0"]:pIdx["v0"]+5]
        rela_q = params[pIdx["rela_q"]:pIdx["rela_q"]+4]
        rela_txyz = self.rela_txyz_lr * params[pIdx["rela_txyz"]:pIdx["rela_txyz"]+3]
        return ex_q, ex_txyz, fx, u0, v0, rela_q, rela_txyz

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

    def getGlobalParamsOfSingleView_as_x0(self, photoType, ex_q, ex_txyz, fx, u0, v0, rela_q, rela_txyz):
        # 生成 scipy.optimize.minimize 函数的部分输入
        ph = photoType.value
        pIdx = {"ex_q":0, "ex_txyz":4, "fx":7, "u0v0":8} # 与x0相对应
        x0 = np.hstack([ex_q[ph], ex_txyz[ph]/self.ex_txyz_lr, \
            fx[ph]/self.fx_lr,  u0[ph]/self.uv_lr, v0[ph]/self.uv_lr])
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            pIdx.update({"rela_q":len(x0), "rela_txyz":len(x0)+4})
            x0 = np.hstack([x0, rela_q, rela_txyz/self.rela_txyz_lr])
        return x0, pIdx

    def getCurrentGlobalParamsOf5Views_as_x0(self):
        pIdx = {"ex_q":0, "ex_txyz":20, "fx":35, "u0":40, "v0":45,\
            "rela_q":50, "rela_txyz":54} # 与x0相对应
        x0 = np.hstack([self.ex_q.flatten(), self.ex_txyz.flatten()/self.ex_txyz_lr, \
            self.fx/self.fx_lr, self.u0/self.uv_lr, self.v0/self.uv_lr, \
            self.rela_q, self.rela_txyz/self.rela_txyz_lr])
        return x0, pIdx

    ###########################################
    ######### Maximization stage 1 ############
    ###########################################

    def residualErrorOfGlobalParams(self, params, pIdx, photoType, withAniScales=False, verbose=True):
        # params = np.array([ex_rx, ex_ry, ex_rz, ex_tx, ex_ty, ex_tz, f, u0, v0])
        ph = photoType.value
        ex_q, ex_txyz, fx, u0, v0, rela_q, rela_txyz = self.parseGlobalParamsOfSingleView(params, pIdx, photoType)
        aniScales = np.array([1.,1.,1.], np.float32)
        errorAniScales = 0.
        gradAniScales = np.zeros((3,),np.float32)
        if withAniScales == True: # 考虑牙列各向异性放缩
            aniScales = params[pIdx["aniScales"]:pIdx["aniScales"]+3] 
            errorAniScales = self.weightAniScale * np.sum((aniScales - 1.)**2) / self.scaleStd**2
            gradAniScales = self.weightAniScale/self.scaleStd**2 * (aniScales - 1.)
        
        rela_R = self.updateRelaRotMat(rela_q)
        extrViewMat = self.updateExtrinsicViewMatrix(ex_q, ex_txyz)
        intrProjMat = self.updateIntrinsicProjectionMatrix(fx, u0, v0)
        tIdx = self.visIdx[ph]
        errorPixel, grad = self.computePixelResidualError(photoType, self.featureVec[tIdx], self.scales[tIdx], self.rotAngleXYZs[tIdx], self.transVecXYZs[tIdx],\
            extrViewMat, intrProjMat, rela_R, rela_txyz, aniScales, stage=1)
        grad[-3:] = grad[-3:] + gradAniScales
        if verbose == True:
            print("errorPixel: {:.2f}, errorAniScales: {:.2f}".format(errorPixel, errorAniScales))
        return errorPixel + errorAniScales, grad



    def residualErrorOfGlobalParamsOf5Views(self, params, pIdx, verbose):
        # params = np.array([ex_ryz, ex_txyz, fx, u0, v0, aniScales])
        ex_q, ex_txyz, fx, u0, v0, rela_q, rela_txyz = self.parseGlobalParamsOf5Views(params, pIdx)
        aniScales = params[pIdx["aniScales"]:pIdx["aniScales"]+3]
        errors = []
        grads = []
        for phType in self.photoTypes:
            sub_x0, sub_pIdx = self.getGlobalParamsOfSingleView_as_x0(phType, ex_q, ex_txyz, fx, u0, v0, rela_q, rela_txyz)
            sub_pIdx["aniScales"] = len(sub_x0)
            sub_x0 = np.hstack([sub_x0, aniScales])
            ph = phType.value
            _sub_error, _sub_grad = self.residualErrorOfGlobalParams(sub_x0, sub_pIdx, phType, withAniScales=True, verbose=verbose)
            sub_error = self.weightViewMaxiStage1[ph] * _sub_error
            sub_grad = self.weightViewMaxiStage1[ph] * _sub_grad
            errors.append(sub_error)
            grads.append(sub_grad)
        grads = np.array(grads)
        grad = np.hstack([grads[:,:4].flatten(), grads[:,4:7].flatten(), grads[:,7], grads[:,8], grads[:,9], np.sum(grads[:,10:], axis=0)])
        if verbose==True:
            print("maximization stage1 step errors: [{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]".format(*errors))
        return np.sum(errors), grad
        
    
    def maximization_stage1_step_5Views(self, method="BFGS", maxiter=10, verbose=True):
        # 优化 ex_q, ex_txyz 与 fx, u0, v0, 以及 rela_q, rela_txyz, aniScales
        x0, pIdx = self.getCurrentGlobalParamsOf5Views_as_x0()
        pIdx["aniScales"] = len(x0)
        x0 = np.hstack([x0, self.aniScales])

        resOptParams = scipy.optimize.minimize(fun=self.residualErrorOfGlobalParamsOf5Views, x0=x0, jac=True,\
            args=(pIdx,False), method=method, options={"maxiter":maxiter})
        params = resOptParams.x

        self.ex_q, self.ex_txyz, self.fx, self.u0, self.v0, self.rela_q, self.rela_txyz =\
            self.parseGlobalParamsOf5Views(params, pIdx)
        self.aniScales = params[pIdx["aniScales"]:pIdx["aniScales"]+3]

        self.loss_maximization_step = self.residualErrorOfGlobalParamsOf5Views(params, pIdx, verbose) # For print purpose



    ###########################################
    ######### Maximization stage 2 ############
    ###########################################

    def anistropicRowScale2ScalesAndTransVecs(self):
        # 将各向异性牙列放缩在stage2之前转化为scale和transVecXYZs
        self.scales = np.prod(self.aniScales)**(1/3) * np.ones_like(self.scales, np.float32)
        self.transVecXYZs = self.X_Mu_centroids * (self.aniScales - 1.)
        self.aniScales = np.array([1.,1.,1.], np.float32)


    def computeTeethPoseResidualError(self, scales, rotAngleXYZs, transVecXYZs, ph, tIdx):
        centeredPoseParams = np.hstack([(transVecXYZs-self.meanTransVecXYZs[tIdx]), (rotAngleXYZs-self.meanRotAngleXYZs[tIdx])]) # shape=(self.numTooth,7)
        centeredScales = scales - self.meanScales[tIdx]
        errorTeethPose = np.sum(np.matmul(np.matmul(centeredPoseParams[:,None,:], self.invCovMats[tIdx,:,:]), centeredPoseParams[:,:,None]))
        errorScales = centeredScales @ self.invCovMatOfScale[tIdx,tIdx[:,None]] @ centeredScales
        return self.weightTeethPose * (errorTeethPose + errorScales)

    
    def residualErrorOfTeethPose(self, params, pIdx, photoType, verbose, step=1):
        ph = photoType.value
        tIdx = self.visIdx[ph]

        ex_q, ex_txyz, fx, u0, v0, rela_q, rela_txyz = self.parseGlobalParamsOfSingleView(params, pIdx, photoType)
        extrViewMat = self.updateExtrinsicViewMatrix(ex_q, ex_txyz)
        intrProjMat = self.updateIntrinsicProjectionMatrix(fx, u0, v0)
        rela_R = self.updateRelaRotMat(rela_q)
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
            self.ex_q, self.ex_txyz, self.fx, self.u0, self.v0, self.rela_q, self.rela_txyz)

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
        self.ex_q[ph], self.ex_txyz[ph], self.fx[ph], self.u0[ph], self.v0[ph], self.rela_q, self.rela_txyz =\
            self.parseGlobalParamsOfSingleView(params, pIdx, photoType)
        self.transVecXYZs[tIdx], self.rotAngleXYZs[tIdx], self.scales[tIdx] = self.parseTeethPoseParams(self, params, pIdx, tIdx, step)

        self.residualErrorOfTeethPose(params, pIdx, photoType, verbose, step) # For print
    
    
    def residualErrorOfTeethPoseOf5Views(self, params, pIdx, verbose, step=1):
        ex_q, ex_txyz, fx, u0, v0, rela_q, rela_txyz = self.parseGlobalParamsOf5Views(params, pIdx)
        tIdx = [i for i in range(self.numTooth)]
        transVecXYZs, rotAngleXYZs, scales = self.parseTeethPoseParams(params, pIdx, tIdx, step)

        errors = []
        for phType in self.photoTypes:
            ph = phType.value
            tIdx = self.visIdx[ph]
            numT = len(tIdx)
            sub_x0, sub_pIdx = self.getGlobalParamsOfSingleView_as_x0(phType, ex_q, ex_txyz, fx, u0, v0, rela_q, rela_txyz)
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


    def maximization_stage2_step_5Views(self, step=1, verbose=True):
        x0, pIdx = self.getCurrentGlobalParamsOf5Views_as_x0()

        rhobeg = 1.0
        maxiter = 1000
        alpha = 0.1

        if step == 1:
            pIdx["tXYZs"] = len(x0)
            x0 = np.hstack([x0, self.transVecXYZs.flatten()])
            rhobeg = alpha * self.transVecStd
        elif step == 2:
            pIdx["rXYZs"] = len(x0)
            x0 = np.hstack([x0, self.rotAngleXYZs.flatten()])
            rhobeg = alpha * self.rotAngleStd
        elif step == 3:
            pIdx["scales"] = len(x0)
            x0 = np.hstack([x0, self.scales])
            rhobeg = alpha * self.scaleStd
        elif step == 4:
            pIdx.update({"tXYZs":len(x0), "rXYZs":len(x0)+self.numTooth*3, "scales":len(x0)+self.numTooth*6})
            x0 = np.hstack([x0, (self.transVecXYZs/self.transVecStd).flatten(), (self.rotAngleXYZs/self.rotAngleStd).flatten(), (self.scales-1.)/self.scaleStd])
        
        optRes = scipy.optimize.minimize(fun=self.residualErrorOfTeethPoseOf5Views, x0=x0, args=(pIdx, False, step), \
            method="COBYLA", options={"rhobeg":rhobeg,"maxiter":maxiter})
        params = optRes.x

        self.ex_q, self.ex_txyz, self.fx, self.u0, self.v0, self.rela_q, self.rela_txyz =\
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
                self.rela_R, self.rela_txyz, aniScales=np.ones((3,),np.float32), stage=3)
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
            canvasShape = self.edgeMask[ph].shape
        canvas = np.zeros((*canvasShape,3), dtype=np.float32)
        h, w = self.edgeMask[ph].shape
        canvas[:h,:w,:] = self.edgeMask[ph][:,:,None] # white: ground truth
        
        edgePred = np.zeros(canvasShape, dtype=np.float32)
        edgePred[self.P_pred[ph][:,1], self.P_pred[ph][:,0]] = 1. # red: edge prediction
        if dilate == True:
            edgePred = skimage.morphology.binary_dilation(edgePred, skimage.morphology.disk(2)) # dilation edge prediction for visualization
        canvas[:,:,0] = np.max(np.stack([edgePred,canvas[:,:,0]]), axis=0)
        
        plt.figure(figsize = (10,10))
        plt.imshow(canvas)