import numpy as np
from copy import copy,deepcopy
import os 
from matplotlib import pyplot as plt
import scipy
from scipy.spatial.transform import Rotation as RR
import skimage
import h5py
import ray
import emopt_func
from projection_utils import PHOTO




class EMOpt5Views(object):
    def __init__(self, edgeMasks, photoTypes, visMasks, Mask, Mu, Mu_normals, SqrtEigVals, Sigma, PoseCovMats, ScaleCovMat, transVecStd, rotVecStd) -> None:
        self.photoTypes = sorted(photoTypes, key=lambda x:x.value)
        assert self.photoTypes == [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
        
        self.edgeMask = [None] * 5 # order as the Enum value in PHOTO
        self.visIdx = [None] * 5 # 每张照片中出现的牙齿轮廓的牙齿的Mask
        for phType, eMask, visMask in zip(photoTypes, edgeMasks, visMasks):
            assert eMask.ndim == 2, "edgeMask should be grayscale" # 单通道Mask图片
            self.edgeMask[phType.value] = eMask # binary 2d-array
            self.visIdx[phType.value] = np.argwhere(visMask[Mask]>0).flatten()

        self.P_true = []
        for v in self.edgeMask:
            p_true = np.argwhere(v>0)[:,::-1].astype(np.float64)
            self.P_true.append(np.ascontiguousarray(p_true, np.float64))
            # num_point_sampled = min(2000, len(p_true))
            # ds_p_true = emopt_func.farthestPointDownSample(p_true, num_point_sampled)
            # self.P_true.append(np.ascontiguousarray(ds_p_true, np.float64))

        self.P_true_normals = [np.ascontiguousarray(emopt_func.initEdgeMaskNormals(v), np.float64) for v in self.P_true]
        self.M = [len(v) for v in self.P_true] # 真实Mask中边缘像素点的数量
        self.flag_valid = [np.ones((m,),np.bool_) for m in self.M]
        self.P_true_valid = [p_true[flag] for flag,p_true in zip(self.flag_valid,self.P_true)]
        
        # 分为上下牙列
        self.Mask = Mask
        Mask_U, Mask_L = np.split(Mask, 2, axis=0)
        self.numUpperTooth = int(np.sum(Mask_U)) #上牙列的牙齿数量
        self.numTooth = int(np.sum(Mask))
        self.numPoint = Mu.shape[1]
        # 记录正视图、左视图、右视图中上下牙列visIdx的区分id
        self.ul_sp = {phType.value:np.argwhere(self.visIdx[phType.value] >= self.numUpperTooth).min() for phType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]}

        # 上牙列index=0 下牙列index=1
        self.X_Mu = Mu[Mask]
        self.X_Mu_normals = Mu_normals[Mask]

        self.X_Mu_centroids = self.X_Mu.mean(axis=1)
        # self.X_Mu_U_Cen =  self.X_Mu_centroids[:self.numUpperTooth].mean(axis=0) # 原点 in world coord
        # self.X_Mu_L_Cen =  self.X_Mu_centroids[self.numUpperTooth:].mean(axis=0) # 原点 in world coord

        self.SqrtEigVals = SqrtEigVals[Mask] # shape=(self.numTooth, 1, numPC)
        self.SigmaT = np.transpose(Sigma[Mask],(0,2,1))

        self.meanRotVecXYZs = np.zeros((self.numTooth,3)) # 每颗牙齿相对旋转角度的均值 # 待讨论
        self.meanTransVecXYZs = np.zeros((self.numTooth,3)) # 每颗牙齿相对平移的均值 # 待讨论
        self.meanScales = np.ones((self.numTooth,)) #每颗牙齿的相对尺寸
        self.invCovMats = np.linalg.inv(PoseCovMats[Mask])
        self.invCovMatOfScale = np.linalg.inv(ScaleCovMat[Mask][:,Mask])

        # init teeth shape subspace
        self.numPC = SqrtEigVals.shape[-1] 
        self.featureVec = np.zeros(self.SqrtEigVals.shape, dtype=np.float64) # shape=(self.numTooth, 1, numPC), mean=0, std=1
        
        # init teeth scales, rotation vecs around X-Y-Z axes, translation vectors along X-Y-Z axes
        self.scales = np.ones((self.numTooth,), np.float64)
        self.rotVecXYZs = np.zeros((self.numTooth,3), np.float64)
        self.transVecXYZs = np.zeros((self.numTooth,3), np.float64)
        
        self.rowScaleXZ = np.ones((2,), dtype=np.float64) # 各向异性牙列放缩, 仅用于maximization stage1, 在stage2之前转化为scale和transVecXYZs


        # init extrinsic param of camera
        self.ex_rxyz_default = {PHOTO.UPPER: np.array([0.7*np.pi, 0., 0.], dtype=np.float64), # upper
                                PHOTO.LOWER: np.array([-0.7*np.pi, 0., 0.], dtype=np.float64), # lower
                                PHOTO.LEFT: np.array([2.80, 0, -1.42], dtype=np.float64), # left
                                PHOTO.RIGHT: np.array([2.80, 0, 1.42], dtype=np.float64),  # right
                                PHOTO.FRONTAL: np.array([np.pi, 0., 0.], dtype=np.float64)  }# frontal
        self.ex_txyz_default = {PHOTO.UPPER: np.array([0., 0., 70.], dtype=np.float64), # upper # 70
                                PHOTO.LOWER: np.array([0., 0., 70.], dtype=np.float64),  # lower # 70
                                PHOTO.LEFT: np.array([-5., 0., 120.], dtype=np.float64), # left # [-5,0,70]
                                PHOTO.RIGHT: np.array([5., 0., 120.], dtype=np.float64),  # right # [5,0,70]
                                PHOTO.FRONTAL: np.array([0., -2., 120.], dtype=np.float64) }  # frontal # [0,-2,70]
        self.ex_rxyz = np.empty((5,3), dtype=np.float64) # shape=(5,3) # init rot vecs around x-y-z axis based on photoType
        self.ex_txyz = np.empty((5,3), dtype=np.float64) # shape=(5,3) # init trans vector
        # init intrinsic param of camera
        self.focLth = np.empty((5,), dtype=np.float64)
        self.dpix = np.empty((5,), dtype=np.float64)
        self.u0 = np.empty((5,), dtype=np.float64)
        self.v0 = np.empty((5,), dtype=np.float64)
        for photoType in self.photoTypes:
            self.initExtrIntrParams(photoType)

        self.rela_rxyz_default = np.array([0.,0.,0.],dtype=np.float64) #下牙列相对于上牙列的旋转
        self.rela_txyz_default = np.array([0.,-5.,0.],dtype=np.float64) #下牙列相对于上牙列的位移
        self.rela_rxyz = None #下牙列相对于上牙列的旋转
        self.rela_R = None
        self.rela_txyz = None #下牙列相对于上牙列的位移
        self.initRelativeToothRowPose()

        # approx. learning step (Not used)
        self.ex_rxyz_lr = 1.
        self.ex_txyz_lr = 1.
        self.focLth_lr = 1.
        self.uv_lr = 1.
        self.dpix_lr = 1.
        self.rela_rxyz_lr = 1. # 0.001
        self.rela_txyz_lr = 1. # 0.1
        
        self.varAngle = 0.09 # param in expectation loss
        self.varPoint = 25.  # param in residual pixel error in maximization loss
        self.varPlane = 0.5  # param in residual pixel error in maximization loss
        # weight in maximization step for 5 views: [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]

        self.weightViews = np.array([1.,1.,1.,1.,1.], dtype=np.float64) # [3,3,1,1,1]
        self.weightAniScale = 1.
        self.weightTeethPose = 1. # param in residual teeth pose error in maximization loss
        self.weightFeatureVec = 1. # param in residual featureVec error in maximization loss
        
        self.transVecStd = transVecStd
        self.scaleStd = np.mean(np.sqrt(np.diag(ScaleCovMat)))
        self.rotVecStd = rotVecStd


        self.X_deformed = self.X_Mu.copy() # np.empty(self.X_Mu.shape, dtype=np.float64)
        self.X_deformed_normals = self.X_Mu_normals.copy() # np.empty(self.X_Mu.shape, dtype=np.float64)
        self.RotMats = np.tile(np.eye(3),(self.numTooth,1,1))
        self.X_trans = self.X_Mu.copy() # np.empty(self.X_Mu.shape, dtype=np.float64)
        self.X_trans_normals = self.X_Mu_normals.copy() # np.empty(self.X_Mu.shape, dtype=np.float64)
        

        self.extrViewMat = np.empty((5,4,3), dtype=np.float64) # homo world coord (xw,yw,zw,1) to camera coord (xc,yc,zc): 4*3 right-multiplying matrix
        self.X_camera = [None] * 5 # compute X in camera coord based on X in world coord, ndarray, shape=(numTooth,1500,3)
        self.X_camera_normals = [None] * 5
        
        self.intrProjMat = np.empty((5,3,3), dtype=np.float64) # camera coord (xc,yc,zc) to image coord (u,v,zc): 3*3 right-multiplying matrix
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

        self.loss_expectation_step = np.zeros((5,))
        self.corre_pred_idx = [None] * 5
        self.loss_maximization_step = 0.





    ###########################################
    ######### Initialization functions ########
    ###########################################


    def initExtrinsicRotVecs(self, photoType):
        ph = photoType.value
        self.ex_rxyz[ph] = self.ex_rxyz_default[photoType].copy()
    
    def initExtrinsicTransVec(self, photoType):
        ph = photoType.value
        self.ex_txyz[ph] = self.ex_txyz_default[photoType].copy()

    def initCameraIntrinsicParams(self, photoType):
        ph = photoType.value
        focLth = {PHOTO.UPPER:100.0, PHOTO.LOWER:100.0, PHOTO.LEFT:100.0, PHOTO.RIGHT:100.0, PHOTO.FRONTAL:100.0} # [50,50,35,35,35]
        dpix = {PHOTO.UPPER:0.1, PHOTO.LOWER:0.1, PHOTO.LEFT:0.06, PHOTO.RIGHT:0.06, PHOTO.FRONTAL:0.06} # 0.06
        self.focLth[ph] = focLth[photoType]
        self.dpix[ph] = dpix[photoType] 
        self.u0[ph] = self.edgeMask[ph].shape[1]/2. # img.width/2
        self.v0[ph] = self.edgeMask[ph].shape[0]/2. # img.height/2

    def initExtrIntrParams(self, photoType):
        self.initExtrinsicRotVecs(photoType)
        self.initExtrinsicTransVec(photoType)
        self.initCameraIntrinsicParams(photoType)

    def initRelativeToothRowPose(self):
        self.rela_rxyz = self.rela_rxyz_default.copy() #下牙列相对于上牙列的旋转
        self.rela_R = emopt_func.rotvec2rotmat(self.rela_rxyz)
        self.rela_txyz = self.rela_txyz_default.copy() #下牙列相对于上牙列的位移


    def updateCameraParams(self, p2d, p3d_lst, phType, rela_txyz, rela_R=np.identity(3)):
        ph = phType.value
        if phType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            p3d_lst = p3d_lst[:ul_sp] + [x@rela_R+rela_txyz for x in p3d_lst[ul_sp:]]
        _exRotMat, ex_t, _intrProjMat_T = emopt_func.solveCameraParamsbyDLT(p2d, np.vstack(p3d_lst))
        self.ex_txyz[ph] = ex_t
        self.ex_rxyz[ph] = RR.from_matrix(_exRotMat).as_rotvec()
        self.focLth[ph] = self.dpix[ph] * (_intrProjMat_T[0,0] + _intrProjMat_T[1,1])/2.
        self.u0[ph] = _intrProjMat_T[0,2]
        self.v0[ph] = _intrProjMat_T[1,2]
        print("Estimate camera params of ", phType)
    

    def assignValue2ExtrParamByName(self, photoType, paramName, value, assign2DefaultValue=False):
        # print(paramName, value)
        # param name should be in ['r.x', 'r.y', 'r.z', 't.x', 't.y', 't.z']
        ph = photoType.value
        xyz2i = {"x":0,"y":1,"z":2,"xyz":[0,1,2]}
        r_t, x_y_z = paramName.split(".")
        i = xyz2i[x_y_z] # i=0,1,2
        if r_t == "r":
            self.ex_rxyz[ph,i] = value # variable
            if assign2DefaultValue == True:
                self.ex_rxyz_default[photoType][i] = value
        elif r_t == "t":
            self.ex_txyz[ph,i] = value # variable
            if assign2DefaultValue == True:
                self.ex_txyz_default[photoType][i] = value
        else:
            print("param name should be in ['r.x', 'r.y', 'r.z', 'r.xyz', 't.x', 't.y', 't.z']")


    def gridSearchExtrinsicParams(self):
        # 对于每张图片，粗略搜索良好的初始条件, 更新时不考虑上下牙列相对位姿
        ExtrParamSearchSpace = {PHOTO.UPPER:{"r.x": np.pi*np.array([0.6, 0.65, 0.7, 0.75, 0.8],np.float64)},
                                PHOTO.LOWER:{"r.x": np.pi*np.array([-0.6, -0.65, -0.7, -0.75, -0.8],np.float64)},
                                PHOTO.LEFT:{"r.xyz": np.array([[3.11, 0, -0.49], [3.05, 0, -0.73], [2.99, 0, -0.97], [2.90, 0, -1.20], [2.80, 0, -1.43]],np.float64)},
                                PHOTO.RIGHT:{"r.xyz": np.array([[3.11, 0, 0.49], [3.05, 0, 0.73], [2.99, 0, 0.97], [2.90, 0, 1.20], [2.80, 0, 1.43]],np.float64)},
                                PHOTO.FRONTAL:{"r.x": np.pi*np.array([0.98, 1., 1.02],np.float64)} }
        self.initRelativeToothRowPose()
        for phType, paramDict in ExtrParamSearchSpace.items():
            ph = phType.value
            for paramName, paramSearchSpace in paramDict.items():
                print(phType, paramName, paramSearchSpace)
                
                P_preds = []
                for paramValue in paramSearchSpace:
                    self.initExtrIntrParams(phType) # init extrinsic and intrinsic camera params with default values
                    self.assignValue2ExtrParamByName(phType, paramName, paramValue)
                    self.updateEdgePrediction(phType) # 更新 X_Mu_pred
                    P_preds.append(self.P_pred[ph])
                TP_preds = ray.get([emopt_func.ray_cpd_rigid_registration_2D.remote(self.P_true[ph], p_pred) for p_pred in P_preds])
                
                losses = []
                for idx,paramValue in enumerate(paramSearchSpace):
                    self.initExtrIntrParams(phType) # init extrinsic and intrinsic camera params with default values
                    self.assignValue2ExtrParamByName(phType, paramName, paramValue)
                    self.updateEdgePrediction(phType) # 更新 X_Mu_pred
                    self.updateCameraParams(TP_preds[idx], self.X_Mu_pred[ph], phType, self.rela_txyz, self.rela_R) # update extrinsic and intrinsic camera params
                    losses.append(self.expectation_step(phType, stage=0, verbose=True, use_percentile=False))
                
                idx_selected = np.argmin(losses)
                bestParamValue = paramSearchSpace[idx_selected] # best guess from expectation loss
                print("Best param guess: ", bestParamValue)
                self.assignValue2ExtrParamByName(phType, paramName, bestParamValue, assign2DefaultValue=True) # update default values with the best guess
                self.initExtrIntrParams(phType) # init extrinsic and intrinsic camera params with default values
                self.updateEdgePrediction(phType) # 更新 X_Mu_pred
                self.updateCameraParams(TP_preds[idx_selected], self.X_Mu_pred[ph], phType, self.rela_txyz, self.rela_R) # update extrinsic and intrinsic camera params
            print("-"*50)
    

    def assignValue2RelaPoseParamByName(self, paramName, value, assign2DefaultValue=False):
        # param name should be in ['rela.r.x', 'rela.r.y', 'rela.r.z', 'rela.t.x', 'rela.t.y', 'rela.t.z']
        print(paramName, value)
        xyz2i = {"x":0,"y":1,"z":2}
        _, r_t, x_y_z = paramName.split(".")
        i = xyz2i[x_y_z] # i=0,1,2
        if r_t == "r":
            self.rela_rxyz[i] = value # variable
            if assign2DefaultValue == True:
                self.rela_rxyz_default[i] = value
        elif r_t == "t":
            self.rela_txyz[i] = value # variable
            if assign2DefaultValue == True:
                self.rela_txyz_default[i] = value
        else:
            print("param name should be in ['rela.r.x', 'rela.r.y', 'rela.r.z', 'rela.t.x', 'rela.t.y', 'rela.t.z']")
            
    
    def gridSearchRelativePoseParams(self):
        # 在粗略搜索良好的初始条件后, 考虑上下牙列相对位姿 rela_rxyz, rela_txyz
        RelativePoseParamSearchSpace = {"rela.t.z": (self.rela_txyz_default[2]+np.array([-3,-2,-1,0,1,2,3]), [PHOTO.LEFT, PHOTO.RIGHT]),# tz:left,right
                            "rela.t.y": (np.array([-7,-6,-5,-4]), [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]), # ty:left,right,frontal
                            "rela.t.x": (np.array([-1.,0.,1.]), [PHOTO.FRONTAL,])} # tx:frontal
        for paramName, (paramSearchSpace, phTypes) in RelativePoseParamSearchSpace.items():
            self.initRelativeToothRowPose()
            num_photo_relevant = len(phTypes)

            P_preds = []
            rela_txyz_list = []
            for paramValue in paramSearchSpace:
                self.assignValue2RelaPoseParamByName(paramName, paramValue) # 初始化牙列相对位姿参数
                for phType in phTypes:
                    ph = phType.value
                    self.initExtrIntrParams(phType) # init extrinsic and intrinsic camera params with default values
                    self.updateEdgePrediction(phType)
                    P_preds.append(self.P_pred[ph])
                    rela_txyz_list.append(self.rela_txyz)
            TP_preds = ray.get([emopt_func.ray_cpd_rigid_registration_2D.remote(self.P_true[ph], p_pred) for p_pred in P_preds])

            losses = []
            for idx, paramValue in enumerate(paramSearchSpace):
                loss = 0.
                self.assignValue2RelaPoseParamByName(paramName, paramValue) # 初始化牙列相对位姿参数
                for jdx, phType in enumerate(phTypes):
                    ph = phType.value
                    self.initExtrIntrParams(phType) # init extrinsic and intrinsic camera params with default values
                    self.updateEdgePrediction(phType)
                    i = idx * num_photo_relevant + jdx
                    self.updateCameraParams(TP_preds[i], self.X_Mu_pred[ph], phType, rela_txyz_list[i]) # update extrinsic and intrinsic camera params
                    loss += self.expectation_step(phType, stage=0, verbose=True, use_percentile=False) 
                losses.append(loss)

            idx_selected = np.argmin(losses)
            bestParamValue = paramSearchSpace[idx_selected] # best guess from expectation loss
            self.assignValue2RelaPoseParamByName(paramName, bestParamValue, assign2DefaultValue=True) # update default values with the best guess
            print("Best param guess: ", bestParamValue)
            for jdx, phType in enumerate(phTypes):
                ph = phType.value
                self.initExtrIntrParams(phType) # init extrinsic and intrinsic camera params with default values
                self.updateEdgePrediction(phType)
                i = idx_selected * num_photo_relevant + jdx
                self.updateCameraParams(TP_preds[i], self.X_Mu_pred[ph], phType, rela_txyz_list[i]) # update extrinsic and intrinsic camera params
            print("-"*50)


    def searchDefaultRelativePoseParams(self):
        # 判断下压列包裹上牙列或上牙列包裹下牙列
        phType = PHOTO.FRONTAL 
        ph = phType.value
        RelativePoseParamSearchSpace = {"rela.t.z":np.array([0,3,6])}
        for paramName, SearchSpace in RelativePoseParamSearchSpace.items():
            P_preds = []
            rela_txyz_list = []
            for paramValue in SearchSpace:
                self.initExtrIntrParams(phType) # init extrinsic and intrinsic camera params with default values
                self.assignValue2RelaPoseParamByName(paramName, paramValue) # 初始化牙列相对位姿参数
                self.updateEdgePrediction(phType) # 更新 X_Mu_pred
                P_preds.append(self.P_pred[ph])
                rela_txyz_list.append(self.rela_txyz)
            TP_preds = ray.get([emopt_func.ray_cpd_rigid_registration_2D.remote(self.P_true[ph], p_pred) for p_pred in P_preds])

            losses = []
            for i,paramValue in enumerate(SearchSpace):
                self.initExtrIntrParams(phType) # init extrinsic and intrinsic camera params with default values
                self.assignValue2RelaPoseParamByName(paramName, paramValue) # 初始化牙列相对位姿参数
                self.updateEdgePrediction(phType) # 更新 X_Mu_pred
                self.updateCameraParams(TP_preds[i], self.X_Mu_pred[ph], phType, rela_txyz_list[i]) # update extrinsic and intrinsic camera params
                losses.append(self.expectation_step(phType, stage=0, verbose=True, use_percentile=False))

            idx_selected = np.argmin(losses)
            bestParamValue = SearchSpace[idx_selected] # best guess from expectation loss
            self.assignValue2RelaPoseParamByName(paramName, bestParamValue, assign2DefaultValue=True) # update default values with the best guess
            print("Best param guess: ", bestParamValue)
            self.initExtrIntrParams(phType) # init extrinsic and intrinsic camera params with default values
            self.updateEdgePrediction(phType)
            self.updateCameraParams(TP_preds[idx_selected], self.X_Mu_pred[ph], phType, rela_txyz_list[idx_selected]) # update extrinsic and intrinsic camera params
        print("-"*50)

    ###############################################
    # Deformatin in shape subspace for each tooth #
    ###############################################

    def updateDeformedPointPos(self, featureVec, tIdx):
        deformField = np.matmul(featureVec*self.SqrtEigVals[tIdx], self.SigmaT[tIdx]) # shape=(numTooth,1,3*self.numPoint)
        return self.X_Mu[tIdx] + deformField.reshape(self.X_Mu[tIdx].shape) # X_deformed

    def updateDeformedPointNomrals(self):
        pass




    ###########################################
    ######### Update in E step ################
    ###########################################
        


    def updateAlignedPointCloudInWorldCoord(self, tIdx, stage=3): 
        '''更新上下牙列世界坐标系中的三维预测(更新涉及的相关参数：牙齿形状，牙齿大小，牙齿对于均值模型相对位姿，上下牙列的相对位姿)
        暂未考虑下牙列相对于上牙列的位姿关系'''
        if stage >= 3:
            self.X_deformed[tIdx] = self.updateDeformedPointPos(self.featureVec[tIdx], tIdx)
            self.X_deformed_normals[tIdx] = emopt_func.computeGroupedPointNormals(self.X_deformed[tIdx])
        if stage >= 2:
            self.RotMats[tIdx] = emopt_func.rotvec2rotmat(self.rotVecXYZs[tIdx])
            self.X_trans[tIdx] = emopt_func.updateTransformedPointPos(self.X_deformed[tIdx], self.scales[tIdx], self.RotMats[tIdx], self.transVecXYZs[tIdx], self.X_Mu_centroids[tIdx])
            self.X_trans_normals[tIdx] = emopt_func.updateTransformedPointNormals(self.X_deformed_normals[tIdx], self.RotMats[tIdx])
        if stage == 1:
            self.X_trans[tIdx] = np.hstack([self.rowScaleXZ[0],1.,self.rowScaleXZ[1]]) * self.X_trans[tIdx] # self.rowScaleXZ = [1.,1.,1.] after maximization stage 1
    
    

    def updateEdgePrediction(self, photoType):
        # 根据拍摄角度，决定使用上牙列或下牙列点云
        ph = photoType.value
        tIdx = self.visIdx[ph]
        X_trans = self.X_trans[tIdx] # upper
        X_trans_normals = self.X_trans_normals[tIdx]

        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            self.rela_R = emopt_func.rotvec2rotmat(self.rela_rxyz)
            X_trans = emopt_func.updatePointPosByRelaPose(X_trans, self.rela_R, self.rela_txyz, ul_sp)  # left, right, frontal
            X_trans_normals = emopt_func.updatePointNormalsByRelaPose(X_trans_normals, self.rela_R, ul_sp) 
        
        self.extrViewMat[ph] = emopt_func.updateExtrinsicViewMatrix(self.ex_rxyz[ph], self.ex_txyz[ph]) # homo world coord (xw,yw,zw,1) to camera coord (xc,yc,zc): 4*3 right-multiplying matrix
        self.X_camera[ph] = emopt_func.updatePointPosInCameraCoord(X_trans, self.extrViewMat[ph]) # compute X in camera coord based on X in world coord, ndarray, shape=(numTooth,1500,3)
        self.X_camera_normals[ph] = emopt_func.updatePointNormalsInCameraCoord(X_trans_normals, self.extrViewMat[ph,:3,:])
        
        self.intrProjMat[ph] = emopt_func.updateIntrinsicProjectionMatrix(self.focLth[ph], self.dpix[ph], self.u0[ph], self.v0[ph]) # camera coord (xc,yc,zc) to image coord (u,v,zc): 3*3 right-multiplying matrix
        self.X_uv[ph] = emopt_func.updatePointPosInImageCoord(self.X_camera[ph], self.intrProjMat[ph]) # compute X in image coord based on X_camera in camera coord, ndarray, shape=(numTooth,1500,2)
        self.X_uv_normals[ph] = emopt_func.updatePointNormalsInImageCoord(self.X_camera_normals[ph])
        
        # visible points in image coord, and corre idx in X
        avg_depth = self.X_camera[ph][...,2].mean(axis=1) # avg_depth: array shape (numTooth,)
        priority = avg_depth.argsort()
        # self.vis_hull_vertices[ph], self.vis_hull_vertex_indices[ph] = emopt_func.extract_visible_edge_points(self.X_uv[ph], priority, bincount=180, fluc=0.9) 
        self.vis_hull_vertices[ph], self.vis_hull_vertex_indices[ph] = emopt_func.extractVisibleEdgePointsByAvgDepth(self.X_uv[ph].astype(np.int64), priority)
        
        self.P_pred[ph] = np.ascontiguousarray(np.vstack(self.vis_hull_vertices[ph]), np.float64)  # edgeMask prediction 2d-array, shape=(?,2)
        self.P_pred_normals[ph] = np.ascontiguousarray(np.vstack([x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_uv_normals[ph], self.vis_hull_vertex_indices[ph])]), np.float64)   # edgeMask normals prediction 2d-array, shape=(?,2)
        
        # X_Mu_pred[ph] 中的某个元素可能为空，即np.array([])
        self.X_Mu_pred[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_Mu[tIdx], self.vis_hull_vertex_indices[ph])] # points in world coord corre to edgeMask prediction
        self.X_Mu_pred_normals[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_Mu_normals[tIdx], self.vis_hull_vertex_indices[ph])]
        self.X_deformed_pred[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_deformed[tIdx], self.vis_hull_vertex_indices[ph])] 
        self.X_deformed_pred_normals[ph] = [x[vis_hull_vids] for x,vis_hull_vids in zip(self.X_deformed_normals[tIdx], self.vis_hull_vertex_indices[ph])] 
        
    
    
    ###########################################
    ######### Update & Expectation Step #######
    ###########################################

    def expectation(self, photoType, verbose, use_percentile=True):
        ph = photoType.value
        _corre_pred_idx, losses = emopt_func.numba_get_global_point_correspondences(self.P_true[ph], self.P_pred[ph], \
            self.P_true_normals[ph], self.P_pred_normals[ph], self.varAngle)
        if use_percentile == True:
            # l1_point_loss_mat = distance_matrix(self.P_true[ph], self.P_pred[ph], p=1, threshold=int(1e8))
            # l1_point_losses = l1_point_loss_mat[np.arange(self.M[ph]), _corre_pred_idx]
            # self.flag_valid[ph] = l1_point_losses < (2.5 * 1.4826 * np.median(l1_point_losses))
            # 99-percentile
            self.flag_valid[ph] = losses < np.percentile(losses, 99.0)
            self.corre_pred_idx[ph] = _corre_pred_idx[self.flag_valid[ph]] # mapping from [0,M] to [0,len(num_pred_point)]
            self.P_true_valid[ph] = self.P_true[ph][self.flag_valid[ph]]
            self.loss_expectation_step[ph] = np.sum(losses[self.flag_valid[ph]])
        else:
            self.corre_pred_idx[ph] = _corre_pred_idx
            self.loss_expectation_step[ph] = np.sum(losses)

        if verbose==True:
            print("{} - unique pred points: {} - E-step loss: {:.2f}".format(str(photoType), len(np.unique(self.corre_pred_idx[ph])), self.loss_expectation_step[ph]))
    
    def expectation_step(self, photoType, stage, verbose=True, use_percentile=True):
        # 根据新的edgePredcition计算对应点对关系
        ph = photoType.value
        self.updateAlignedPointCloudInWorldCoord(self.visIdx[ph], stage)
        self.updateEdgePrediction(photoType)
        self.expectation(photoType, verbose, use_percentile)
        return self.loss_expectation_step[ph]

    def expectation_step_5Views(self, stage, verbose=True):
        # 根据新的edgePredcition计算对应点对关系,对5张图同时进行
        tIdx = [i for i in range(self.numTooth)]
        self.updateAlignedPointCloudInWorldCoord(tIdx, stage)
        for photoType in self.photoTypes:
            self.updateEdgePrediction(photoType)
            self.expectation(photoType, verbose)

    

    def save_expectation_step_result_with_XRef(self, matFileName, X_Ref):
        scipy.io.savemat(matFileName, {"np_invCovMatOfPose":self.invCovMats, "np_invCovMatOfScale":self.invCovMatOfScale,
                        "np_ex_rxyz":self.ex_rxyz, "np_ex_txyz":self.ex_txyz, "np_focLth":self.focLth, "np_dpix":self.dpix, 
                        "np_u0":self.u0, "np_v0":self.v0, "np_rela_rxyz":self.rela_rxyz, "np_rela_txyz":self.rela_txyz, "np_rowScaleXZ":self.rowScaleXZ, 
                        "np_scales":self.scales, "np_rotVecXYZs":self.rotVecXYZs, "np_transVecXYZs":self.transVecXYZs,
                        "np_X_Mu":self.X_Mu, "np_X_Mu_pred":self.X_Mu_pred, "np_X_Mu_pred_normals":self.X_Mu_pred_normals,
                        "np_visIdx":self.visIdx, "np_corre_pred_idx":self.corre_pred_idx, "np_P_true":self.P_true_valid, "np_X_ref":X_Ref})
    


    def load_expectation_step_result(self, filename, stage):
        # 读取matlab运行后得到的MStep的结果，并更新相关参数
        xOptDict = scipy.io.loadmat(filename, squeeze_me=True)
        self.ex_rxyz = xOptDict["np_ex_rxyz"]
        self.ex_txyz = xOptDict["np_ex_txyz"]
        self.focLth = xOptDict["np_focLth"]
        self.dpix = xOptDict["np_dpix"]
        self.u0 = xOptDict["np_u0"]
        self.v0 = xOptDict["np_v0"]
        self.rela_rxyz = xOptDict["np_rela_rxyz"]
        self.rela_txyz = xOptDict["np_rela_txyz"]
        if stage == 0:
            self.rowScaleXZ = np.ones((2,))
        elif stage == 1:
            self.rowScaleXZ = xOptDict["np_rowScaleXZ"]
        elif stage == 2:
            self.scales = xOptDict["np_scales"]
            self.rotVecXYZs = xOptDict["np_rotVecXYZs"]
            self.transVecXYZs = xOptDict["np_transVecXYZs"]



    
    ###########################################
    ######### Visualization ###################
    ###########################################

    
    # def showEdgeMaskPredictionWithGroundTruth(self, photoType, canvasShape=None, dilate=True):
    #     # red: prediction, white: ground truth
    #     ph = photoType.value
    #     if not bool(canvasShape):
    #         canvasShape = self.edgeMask[ph].shape
    #     canvas = np.zeros((*canvasShape,3), dtype=np.float64)
    #     h, w = self.edgeMask[ph].shape
    #     canvas[:h,:w,:] = self.edgeMask[ph][:,:,None] # white: ground truth
        
    #     edgePred = np.zeros(canvasShape, dtype=np.float64)
    #     pix_pred = self.P_pred[ph].astype(np.int64)
    #     edgePred[pix_pred[:,1], pix_pred[:,0]] = 1. # red: edge prediction
    #     if dilate == True:
    #         edgePred = skimage.morphology.binary_dilation(edgePred, skimage.morphology.disk(2)) # dilation edge prediction for visualization
    #     canvas[:,:,0] = np.max(np.stack([edgePred,canvas[:,:,0]]), axis=0)
        
    #     plt.figure(figsize = (10,10))
    #     plt.imshow(canvas)
    #     return canvas
    
    def showEdgeMaskPredictionWithGroundTruth(self, photoType, canvasShape=None, dilate=True):
        # red: prediction, white: ground truth
        ph = photoType.value
        if not bool(canvasShape):
            canvasShape = self.edgeMask[ph].shape
        canvas = np.zeros((*canvasShape,3), dtype=np.float64)
        edgeTrue = np.zeros(canvasShape, dtype=np.float64)
        pix_true = self.P_true[ph].astype(np.int32)
        edgeTrue[pix_true[:,1], pix_true[:,0]] = 1. # white: ground truth
        if dilate == True:
            edgeTrue = skimage.morphology.binary_dilation(edgeTrue, skimage.morphology.disk(2)) # dilation edge prediction for visualization
        canvas[:,:,0] = edgeTrue
        canvas[:,:,1] = edgeTrue
        canvas[:,:,2] = edgeTrue

        edgePred = np.zeros(canvasShape, dtype=np.float64)
        pix_pred = self.P_pred[ph].astype(np.int32)
        edgePred[pix_pred[:,1], pix_pred[:,0]] = 1. # red: edge prediction
        if dilate == True:
            edgePred = skimage.morphology.binary_dilation(edgePred, skimage.morphology.disk(2)) # dilation edge prediction for visualization
        canvas[:,:,0] = np.max(np.stack([edgePred,canvas[:,:,0]]), axis=0)

        plt.figure(figsize = (10,10))
        plt.axis('off')
        plt.imshow(canvas)
        return canvas




    ############################################
    ######### Maximization Step By Python ######
    ############################################

    def anistropicRowScale2ScalesAndTransVecs(self):
        # 将各向异性牙列放缩在stage2之前转化为scale和transVecXYZs
        self.scales = np.prod(self.rowScaleXZ)**(1/3) * np.ones_like(self.scales, np.float64)
        self.transVecXYZs[:,[0,2]] = self.X_Mu_centroids[:,[0,2]] * (self.rowScaleXZ - 1.)
        self.rowScaleXZ = np.array([1.,1.], np.float64)
    
  
    def computePixelResidualError(self, photoType, featureVec, scales, rotVecXYZs, transVecXYZs, extrViewMat, intrProjMat,\
        rela_R, rela_txyz, rowScaleXZ=np.ones((2,),np.float64), stage=1, step=-1, return_grad=False):
        # self.X_?_pred: List of array of points in Mu teeth shape, [ndarray1, ndarray2, ...]
        # self.corre_pred_idx: corre indices after vertically stacking the transformed self.X_?_pred
        
        ph = photoType.value
        tIdx = self.visIdx[ph]
        _corre_pred_idx = self.corre_pred_idx[ph]
        X_deformed_pred = self.X_deformed_pred[ph]
        _X_trans_pred = self.X_deformed_pred[ph]
        X_deformed_pred_normals = self.X_deformed_pred_normals[ph]
        _X_trans_pred_normals = self.X_deformed_pred_normals[ph]

        if stage >= 3: # 考虑shape subspace 的形变
            # X_deformed = self.updateDeformedPointPos(featureVec, tIdx) # ul may be 0,1,-1 # 轮廓点对应原始点云进行Shape subspace变形操作
            # X_deformed_normals = emopt_func.computePointNormals(X_deformed)
            # X_deformed_pred = [x[vis_hull_vids] for x,vis_hull_vids in zip(X_deformed, self.vis_hull_vertex_indices[ph])]
            # X_deformed_pred_normals = [x[vis_hull_vids] for x,vis_hull_vids in zip(X_deformed_normals, self.vis_hull_vertex_indices)]
            X_deformed_pred = [x_mu_pred + np.reshape(sqrtEigVal*fVec@sigmaTseg, x_mu_pred.shape) for x_mu_pred,sqrtEigVal,fVec,sigmaTseg in \
                              zip(self.X_Mu_pred[ph], self.SqrtEigVals[tIdx], featureVec, self.SigmaT_segs[ph])] # 轮廓点对应原始点云进行Shape subspace变形操作
            X_deformed_pred_normals = self.X_deformed_pred_normals[ph]
        
        if stage >= 2: # 考虑每颗牙齿的相对位姿和尺寸
            rotMats = emopt_func.rotvec2rotmat(rotVecXYZs)
            _X_trans_pred = [s*np.matmul(x-tc,R)+t+tc for x,s,R,t,tc in zip(X_deformed_pred, scales, rotMats, transVecXYZs, self.X_Mu_centroids[tIdx])] # 轮廓点对应原始点云按牙齿分别进行缩放刚性变换
            _X_trans_pred_normals = [np.matmul(xn,R) for xn,R in zip(X_deformed_pred_normals, rotMats)]
            
        
        # 需要考虑上下牙列位置关系，对下牙列的点进行相对旋转和平移
        X_trans_pred = deepcopy(_X_trans_pred)
        X_trans_pred_normals = deepcopy(_X_trans_pred_normals)
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            X_trans_pred = X_trans_pred[:ul_sp] + [x@rela_R+rela_txyz for x in X_trans_pred[ul_sp:]]
            X_trans_pred_normals = X_trans_pred_normals[:ul_sp] + [xn@rela_R for xn in X_trans_pred_normals[ul_sp:]]
        
        _X_corre_pred = np.vstack(X_trans_pred)[_corre_pred_idx]
        X_corre_pred = _X_corre_pred.copy()            
        if stage == 1: # 在优化相机参数同时，优化牙列的Anistropic scales
            X_corre_pred = np.hstack([rowScaleXZ[0],1.,rowScaleXZ[1]]) * _X_corre_pred
        X_corre_pred_normals = np.vstack(X_trans_pred_normals)[_corre_pred_idx]

        X_cam_corre_pred = emopt_func.updatePointPosInCameraCoord(X_corre_pred, extrViewMat) #相机坐标系下对应点坐标
        X_cam_corre_pred_normals = emopt_func.updatePointNormalsInCameraCoord(X_corre_pred_normals, extrViewMat[:3,:]) # extrViewMat.shape = (4,3)
        
        P_corre_pred = emopt_func.updatePointPosInImageCoord(X_cam_corre_pred, intrProjMat)
        P_corre_pred_normals = emopt_func.updatePointNormalsInImageCoord(X_cam_corre_pred_normals)
        
        errorVecUV = self.P_true_valid[ph] - P_corre_pred # ci - \hat{ci}
        _M = len(self.P_true_valid[ph])
        resPointError = np.sum(np.linalg.norm(errorVecUV, axis=1)**2) / self.varPoint
        resPlaneError = np.sum(np.sum(errorVecUV*P_corre_pred_normals, axis=1)**2) / self.varPlane
        # print("resPointError:{:.4f}, resPlaneError:{:.4f}".format(resPointError/_M, resPlaneError/_M))
        loss = (resPointError + resPlaneError) / _M
        if not return_grad:
            return loss, None
        
        # 计算loss关于hat_ci和hat_ni梯度
        ci_hatci = errorVecUV # shape=(_M, 2)
        hatni = P_corre_pred_normals # shape=(_M, 2)
        ci_hatci_dot_hatni = np.sum(ci_hatci*hatni, axis=1)
        par_loss_par_hatci = -2./ _M * np.matmul(ci_hatci[:,None,:], \
            (1./self.varPoint * np.identity(2,np.float64) + 1./self.varPlane * np.matmul(hatni[:,:,None],hatni[:,None,:]))) #(_M, 1, 2)
        par_loss_par_hatni = 2./(self.varPlane*_M) * ci_hatci_dot_hatni[:,None,None] * ci_hatci.reshape(_M,1,2)  #(_M, 1, 2)

        g = X_cam_corre_pred # 3d-point after global transformation (_M,3)
        gn = X_cam_corre_pred_normals
        gz = g[:,[2]]
        gxgy_gz = g[:,:2] / gz # (_M,2)
        par_hatci_par_fx = 1./self.dpix[ph] * gxgy_gz[...,None] # (_M,2,1)
        par_hatci_par_u0 = np.array([[1.],[0.]],np.float64)
        par_hatci_par_v0 = np.array([[0.],[1.]],np.float64)

        # 对于相机内参的梯度
        grad_fx = np.sum(np.matmul(par_loss_par_hatci, par_hatci_par_fx))
        grad_dpix = 0
        grad_u0 = np.sum(np.matmul(par_loss_par_hatci, par_hatci_par_u0))
        grad_v0 = np.sum(np.matmul(par_loss_par_hatci, par_hatci_par_v0))

        fx = intrProjMat[0,0]
        par_hatci_par_g = fx/gz[:,:,None] * np.concatenate([np.tile(np.identity(2,np.float64),(_M,1,1)), -gxgy_gz[...,None]],axis=-1) # (_M,2,3)
        par_loss_par_g = np.matmul(par_loss_par_hatci, par_hatci_par_g) # (_M,1,3)
        # par_g_par_ext = np.identity(3,np.float64) 
        par_g_par_exr = -emopt_func.skewMatrices(g)

        # 对于ex_txyz, ex_rxyz的梯度
        grad_ext = np.sum(par_loss_par_g, axis=0) # (1,3)
        par_hatni_par_gn = self.jacobs_hatni_wrt_gn(gn) # (_M,2,3)
        par_loss_par_gn = np.matmul(par_loss_par_hatni, par_hatni_par_gn)  # (_M,1,3)
        par_gn_par_exr = -emopt_func.skewMatrices(gn)
        grad_exr = np.sum(np.matmul(par_loss_par_g, par_g_par_exr), axis=0) + \
             np.sum(np.matmul(par_loss_par_gn, par_gn_par_exr), axis=0) # (1,3)

        R_global = extrViewMat[:3,:].T
        # 计算牙列相对位置关系参数的梯度
        rowScaleMat = np.diag([rowScaleXZ[0],1.,rowScaleXZ[1]])
        grad_relar = np.zeros((1,3),np.float64)
        grad_relat = np.zeros((1,3),np.float64)
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            ks = np.sum([len(x) for x in _X_trans_pred[:ul_sp]]).astype(np.int64)
            idx_l = _corre_pred_idx >= ks # corre_pred_idx中属于下牙列的点的mask
            
            pl = _X_corre_pred[idx_l]   # 下牙列经过相对旋转位移后未经过牙列缩放的点坐标
            pln = X_corre_pred_normals[idx_l]
            par_loss_par_gl = par_loss_par_g[idx_l]
            par_loss_par_gln = par_loss_par_gn[idx_l]
            par_gl_par_relar = np.matmul(R_global@rowScaleMat, -emopt_func.skewMatrices(pl))
            par_gln_par_relar = np.matmul(R_global, -emopt_func.skewMatrices(pln))
            par_gl_par_relat = R_global
            grad_relar = np.sum(np.matmul(par_loss_par_gl, par_gl_par_relar), axis=0) + np.sum(np.matmul(par_loss_par_gln, par_gln_par_relar), axis=0) # (1,3)
            grad_relat = np.sum(np.matmul(par_loss_par_gl, par_gl_par_relat), axis=0) # (1,3)

        _Regularizer = 1e-3
        grad = np.hstack([_Regularizer*np.squeeze(grad_exr), np.squeeze(grad_ext), grad_fx, grad_dpix, grad_u0, grad_v0, _Regularizer*np.squeeze(grad_relar), np.squeeze(grad_relat)])
        
        
        if stage == 1: # 对于牙列各向异性放缩的梯度
            p = _X_corre_pred # stage1:牙列放缩前的世界坐标系中的坐标
            par_g_par_rowScaleXZ = np.matmul(R_global, self.diagMatrices(p)[...,[0,2]]) # (_M,3,2)
            grad_rowScaleXZ = np.sum(np.matmul(par_loss_par_g, par_g_par_rowScaleXZ), axis=0) # (1,2)
            grad = np.hstack([grad, np.squeeze(grad_rowScaleXZ)])
        
        elif stage == 2: # 无牙列缩放，对于每颗牙齿sim(3)的梯度
            numT = self.numTooth
            tIdx = self.visIdx[ph]
            qt_list = [x-tc for x,tc in zip(_X_trans_pred,self.X_Mu_centroids[tIdx])] # list of np.array 下牙列相对于上牙列变换之前的牙齿局部坐标系坐标 qt -tc
            qtn_list = _X_trans_pred_normals
            qt = np.vstack(qt_list)[_corre_pred_idx]
            qtn = np.vstack(qtn_list)[_corre_pred_idx]
            _grad_txyzs = np.zeros((numT,3))
            _grad_rxyzs = np.zeros((numT,3))
            grad_scales = np.zeros((numT,))
            # par_pu_par_qu = np.eye(3); par_pl_par_ql = rela_R.T # 近似单位阵
            # par_pnu_par_qnu = np.eye(3); par_pln_par_qln = rela_R.T # 近似单位阵
            assert len(tIdx)==len(qt_list), "Num of visible teeth should be equal"
            ks = 0
            kt = 0
            for j,tId in enumerate(tIdx):
                ks = copy(kt)
                kt += len(qt_list[j])
                if ks==kt: continue
                mask_j = np.logical_and(_corre_pred_idx>=ks, _corre_pred_idx<kt)
                par_loss_par_pj = np.matmul(par_loss_par_g[mask_j],R_global) #(?,1,3)
                par_loss_par_pnj = np.matmul(par_loss_par_gn[mask_j],R_global) #(?,1,3)

                if step == 1 or step == 4:
                    # par_qj_par_txyzj = np.identity(3)
                    _grad_txyzs[tId] = par_loss_par_pj.sum(axis=0)
                if step == 2 or step == 4:
                    par_qj_par_rxyzj = -emopt_func.skewMatrices(qt[mask_j]) #(?,3,3)
                    par_qnj_par_rxyzj = -emopt_func.skewMatrices(qtn[mask_j]) #(?,3,3)
                    _grad_rxyzs[tId] = np.matmul(par_loss_par_pj, par_qj_par_rxyzj).sum(axis=0) + \
                        np.matmul(par_loss_par_pnj, par_qnj_par_rxyzj).sum(axis=0)
                if step == 3 or step == 4:
                    par_qj_par_scalej = qt[mask_j].reshape(-1,3,1)
                    grad_scales[tId] = np.matmul(par_loss_par_pj, par_qj_par_scalej).sum()
            
            if step == 1:
                grad = np.hstack([grad, _grad_txyzs.flatten()])
            elif step == 2:
                grad = np.hstack([grad, _grad_rxyzs.flatten()])
            elif step == 3:
                grad = np.hstack([grad, grad_scales])
            elif step == 4:
                grad = np.hstack([grad, _grad_txyzs.flatten(), _grad_rxyzs.flatten(), grad_scales])

        elif stage == 3: # 对于每颗牙齿shape vector的梯度
            numT = self.numTooth
            tIdx = self.visIdx[ph]
            qs_list = X_deformed_pred # 形变后的点的坐标
            rotMats = emopt_func.rotvec2rotmat(rotVecXYZs)
            # par_g_par_p = R_global # par_p_par_qt = np.eye(3) # 近似单位阵 # par_qt_par_qs = scales * rotMats
            par_g_par_qs = scales[:,None,None] * np.matmul(R_global, rotMats) # shape=(len(tIdx),3,3) # par_g_par_qs = par_g_par_p @ par_p_par_qt @ par_qt_par_qs
            _grad_fVecs = np.zeros((self.numTooth, self.numPC))
            assert len(tIdx)==len(qs_list), "Num of visible teeth should be equal"
            ks = 0
            kt = 0
            for j,tId in enumerate(tIdx):
                ks = copy(kt)
                kt += len(qs_list[j])
                if ks==kt: continue
                mask_j = np.logical_and(_corre_pred_idx>=ks, _corre_pred_idx<kt)
                par_loss_par_qsj = np.matmul(par_loss_par_g[mask_j], par_g_par_qs[j]) #(len(idx_j),1,3)
                idx_j = _corre_pred_idx[mask_j] - ks
                sqrtEigVals = np.squeeze(self.SqrtEigVals[tId]) #(numPC,)
                corre_sigmaT_seg = self.SigmaT_segs[ph][j].reshape(self.numPC, -1, 3)[:,idx_j,:,None] # (numPC, 3*m(j)) -> (numPC, len(idx_j), 3, 1) # m(j) num of pred visible points in tooth-mesh-j 
                par_loss_par_fVec = sqrtEigVals * np.squeeze(np.matmul(par_loss_par_qsj[None,...], corre_sigmaT_seg)).sum(axis=-1)
                _grad_fVecs[tId] = par_loss_par_fVec
            grad = np.hstack([grad, _grad_fVecs.flatten()])
        return loss, grad


    @staticmethod
    def diagMatrices(a):
        # a: 2d-array, shape=(?,3)
        n, _ = a.shape
        vec_0 = np.zeros((n,1),np.float64)
        vec_a1, vec_a2, vec_a3 = np.split(a, 3, axis=-1)
        return np.stack([vec_a1, vec_0, vec_0, vec_0, vec_a2, vec_0, vec_0, vec_0, vec_a3], axis=-1).reshape((n,3,3))

    @staticmethod
    def jacobs_hatni_wrt_gn(vec_gn):
        # vec_gn.shape = (m, 3), a list of point normals
        m = len(vec_gn)
        vec_gnx = vec_gn[:,0]
        vec_gny = vec_gn[:,1]
        vec_0 = np.zeros_like(vec_gnx, np.float64)
        vec_gnx_gny = vec_gnx * vec_gny
        vec_norm_gnxy = np.linalg.norm(vec_gn[:,:2], axis=1, keepdims=True)
        _jacob = np.stack([vec_gny**2, -vec_gnx_gny, vec_0, vec_gnx**2, -vec_gnx_gny, vec_0], axis=-1).reshape(m, 2, 3)
        return 1./(vec_norm_gnxy[:,:,None]**3) * _jacob




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


    def parseTeethPoseParams(self, params, pIdx, step):
        transVecXYZs = self.transVecXYZs
        rotVecXYZs = self.rotVecXYZs
        scales = self.scales
        numT = self.numTooth
        if step == 1:
            transVecXYZs = params[pIdx["tXYZs"]:pIdx["tXYZs"]+numT*3].reshape(numT, 3)
        elif step == 2:
            rotVecXYZs = params[pIdx["rXYZs"]:pIdx["rXYZs"]+numT*3].reshape(numT, 3)
        elif step == 3:
            scales = params[pIdx["scales"]:pIdx["scales"]+numT]
        elif step == 4:
            transVecXYZs = params[pIdx["tXYZs"]:pIdx["tXYZs"]+numT*3].reshape(numT, 3)
            rotVecXYZs = params[pIdx["rXYZs"]:pIdx["rXYZs"]+numT*3].reshape(numT, 3)
            scales = params[pIdx["scales"]:pIdx["scales"]+numT]
        return transVecXYZs, rotVecXYZs, scales



    def getCurrentGlobalParamsOf5Views_as_x0(self, stage, step):
        # stage 0
        pIdx = {"ex_rxyz":0, "ex_txyz":15, "focLth":30, "dpix":35, "u0":40, "v0":45,\
            "rela_rxyz":50, "rela_txyz":53} # 与x0相对应
        x0 = np.hstack([self.ex_rxyz.flatten()/self.ex_rxyz_lr, self.ex_txyz.flatten()/self.ex_txyz_lr, \
            self.focLth/self.focLth_lr, self.dpix/self.dpix_lr, self.u0/self.uv_lr, self.v0/self.uv_lr, \
            self.rela_rxyz/self.rela_rxyz_lr, self.rela_txyz/self.rela_txyz_lr])
        # stage 1
        if stage == 1: # 优化全局参数和牙列尺寸
            pIdx["rowScaleXZ"] = len(x0)
            x0 = np.hstack([x0, self.rowScaleXZ])
        # stage 2
        elif stage == 2:
            if step == 1:
                pIdx["tXYZs"] = len(x0)
                x0 = np.hstack([x0, self.transVecXYZs.flatten()])
            elif step == 2:
                pIdx["rXYZs"] = len(x0)
                x0 = np.hstack([x0, self.rotVecXYZs.flatten()])
            elif step == 3:
                pIdx["scales"] = len(x0)
                x0 = np.hstack([x0, self.scales])
            elif step == 4:
                pIdx.update({"tXYZs":len(x0), "rXYZs":len(x0)+self.numTooth*3, "scales":len(x0)+self.numTooth*6})
                x0 = np.hstack([x0, self.transVecXYZs.flatten(), self.rotVecXYZs.flatten(), self.scales])
        elif stage == 3:
            pIdx["featureVec"] = len(x0)
            x0 = np.hstack([x0, self.featureVec.flatten()])
        return x0, pIdx


    # 计算牙齿位姿的损失 negative log likelihood
    def computeTeethPoseResidualError(self, scales, rotVecXYZs, transVecXYZs, tIdx, return_grad=False):
        centeredPoseParams = np.hstack([(transVecXYZs-self.meanTransVecXYZs[tIdx]), (rotVecXYZs-self.meanRotVecXYZs[tIdx])]) # shape=(len(tIdx),6)
        A = self.invCovMats[tIdx,:,:] # shape=(len(tIdx),6,6); A = A.T
        x = centeredPoseParams[:,:,None] # shape=(len(tIdx),6,1)
        x_T = np.transpose(x,(0,2,1)) # shape=(len(tIdx),1,6)
        x_T_times_A = np.matmul(x_T, A) # shape=(len(tIdx),1,6)
        errorTeethPose = np.sum(np.matmul(x_T_times_A, x))

        centeredScales = scales - self.meanScales[tIdx]
        B = self.invCovMatOfScale[tIdx,tIdx[:,None]] # shape=(len(tIdx),len(tIdx)); B = B.T
        y = centeredScales # shape=(len(tIdx),)
        y_T_times_B = y @ B
        errorScales = y_T_times_B @ y 
        if not return_grad:
            return self.weightTeethPose*(errorTeethPose+errorScales), None
        # 计算teethPoseError关于tXYZs,rXYZs,scales的梯度
        numT = self.numTooth
        _grad_txyzs = np.zeros((numT,3),np.float64)
        _grad_rxyzs = np.zeros((numT,3),np.float64)
        grad_scales = np.zeros((numT,),np.float64)
        _grad_txyzs[tIdx] = 2. * np.squeeze(x_T_times_A)[:,0:3]
        _grad_rxyzs[tIdx] = 2. * np.squeeze(x_T_times_A)[:,3:6]
        grad_scales[tIdx] = 2. * y_T_times_B
        grad = self.weightTeethPose*np.hstack([_grad_txyzs.flatten(), _grad_rxyzs.flatten(), grad_scales])
        return self.weightTeethPose*(errorTeethPose+errorScales), grad
        
    # 计算形状向量的损失 negative log likelihood
    def computeFeatureVecResidualError(self, featureVec, tIdx, return_grad=False):
        featureVecError = self.weightFeatureVec * np.sum(featureVec[tIdx]**2)
        if not return_grad:
            return featureVecError, None
        _featureVecGrad = np.zeros(featureVec.shape) # (self.numTooth,1,self.numPC)
        _featureVecGrad[tIdx] = 2. * self.weightFeatureVec * featureVec[tIdx]
        return featureVecError, _featureVecGrad.flatten()


    def MStepLoss(self, params, pIdx, stage, step, verbose, return_grad=False):
        ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz = self.parseGlobalParamsOf5Views(params, pIdx)
        rowScaleXZ = np.ones((2,))
        transVecXYZs, rotVecXYZs, scales = self.parseTeethPoseParams(params, pIdx, step)
        ex_R = emopt_func.rotvec2rotmat(ex_rxyz)
        rela_R = emopt_func.rotvec2rotmat(rela_rxyz)
        extrViewMats = np.concatenate([ex_R, ex_txyz[:,None,:]], axis=1)
        featureVec = self.featureVec
        
        aniScaleError = 0.
        if stage == 1:
            rowScaleXZ = params[pIdx["rowScaleXZ"]:pIdx["rowScaleXZ"]+2]
            # rowScaleXYZ = np.array([rowScaleXZ[0], 1., rowScaleXZ[1]])
            # scales = np.prod(rowScaleXYZ)**(1/3) * np.ones((self.numTooth,))
            # rotVecXYZs = np.zeros((self.numTooth,3))
            # transVecXYZs = self.X_Mu_centroids * (rowScaleXYZ - 1.)
            equiva_scale = np.prod(rowScaleXZ)**(1/3)
            aniScaleError = self.weightAniScale * (equiva_scale-1.)**2 / self.scaleStd**2
        elif stage == 3:
            featureVec = params[pIdx["featureVec"]:pIdx["featureVec"]+self.numTooth*self.numPC].reshape(self.featureVec.shape)

        errors = np.zeros((5,))
        paramNum = len(params)
        M_grad = np.zeros((paramNum,))
        if return_grad == True and stage == 1:
            gradRowScaleXZ = 2.*self.weightAniScale / (3.*self.scaleStd**2) * np.prod(rowScaleXZ)**(-1/3) * np.array([rowScaleXZ[1],rowScaleXZ[0]])
            M_grad[pIdx["rowScaleXZ"]:pIdx["rowScaleXZ"]+2] += np.sum(self.weightViews)*gradRowScaleXZ
        
        for phType in self.photoTypes:
            ph = phType.value
            tIdx = self.visIdx[ph]
            intrProjMat = emopt_func.updateIntrinsicProjectionMatrix(focLth[ph], dpix[ph], u0[ph], v0[ph])
            pixelError, pixelGrad = self.computePixelResidualError(phType, featureVec[tIdx], scales[tIdx], rotVecXYZs[tIdx], transVecXYZs[tIdx],\
                extrViewMats[ph], intrProjMat, rela_R, rela_txyz, rowScaleXZ, stage, step, return_grad)
            teethPoseError = 0.
            teethPoseGrad = np.zeros((7*self.numTooth,), np.float64)
            featureVecError = 0.
            featureVecGrad = np.zeros((self.numPC*self.numTooth,), np.float64)
            if stage == 2:
                teethPoseError, teethPoseGrad = self.computeTeethPoseResidualError(scales[tIdx], rotVecXYZs[tIdx], transVecXYZs[tIdx], tIdx, return_grad)
            elif stage == 3:
                featureVecError, featureVecGrad = self.computeFeatureVecResidualError(featureVec, tIdx, return_grad)
            if verbose == True:
                print("{}, pixelError:{:.2f}, teethPoseError:{:.2f}, featureVecError: {:.2f}".format(\
                    str(phType), pixelError, teethPoseError, featureVecError))
            errors[ph] = self.weightViews[ph] * (pixelError + teethPoseError + aniScaleError + featureVecError)
            if return_grad == True:
                M_grad = self.updateMStepGradVector(M_grad, pIdx, self.weightViews[ph]*pixelGrad, \
                    self.weightViews[ph]*teethPoseGrad, self.weightViews[ph]*featureVecGrad, ph, stage, step)
                # print("gradient of {}".format(str(phType)), self.weightViews[ph]*pixelGrad)

        M_loss = np.sum(errors)
        if verbose==True:
            print("maximization step errors: [{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]".format(*errors))
        if not return_grad:
            return M_loss
        return M_loss, M_grad

    def updateMStepGradVector(self, aggGrad, pIdx, pixelGrad, teethPoseGrad, featureVecGrad, ph, stage, step=-1):
        grad_exr = pixelGrad[0:3]
        grad_ext = pixelGrad[3:6]
        grad_fx, grad_dpix, grad_u0, grad_v0 = pixelGrad[6:10]
        grad_relar = pixelGrad[10:13]
        grad_relat = pixelGrad[13:16]

        ex_rxyz_ks = pIdx["ex_rxyz"] + 3*ph
        ex_txyz_ks = pIdx["ex_txyz"] + 3*ph
        ex_focLth_ks = pIdx["focLth"] + ph
        ex_dpix_ks = pIdx["dpix"] + ph
        ex_u0_ks = pIdx["u0"] + ph
        ex_v0_ks = pIdx["v0"] + ph
        rela_rxyz_ks = pIdx["rela_rxyz"]
        rela_txyz_ks = pIdx["rela_txyz"]

        aggGrad[ex_rxyz_ks:ex_rxyz_ks+3] += grad_exr
        aggGrad[ex_txyz_ks:ex_txyz_ks+3] += grad_ext
        aggGrad[ex_focLth_ks] += grad_fx
        aggGrad[ex_dpix_ks] += grad_dpix
        aggGrad[ex_u0_ks] += grad_u0
        aggGrad[ex_v0_ks] += grad_v0
        aggGrad[rela_rxyz_ks:rela_rxyz_ks+3] += grad_relar
        aggGrad[rela_txyz_ks:rela_txyz_ks+3] += grad_relat

        supp_ks = 16 # 额外参数的起始index
        if stage == 1:
            rowScale_ks = pIdx["rowScaleXZ"]
            grad_rowScaleXZ = pixelGrad[supp_ks:supp_ks+2]
            aggGrad[rowScale_ks:rowScale_ks+2] += grad_rowScaleXZ
        elif stage == 2:
            numT = self.numTooth
            if step == 1:
                txyzs_ks = pIdx["tXYZs"]
                aggGrad[txyzs_ks:txyzs_ks+3*numT] += pixelGrad[supp_ks:supp_ks+3*numT] + teethPoseGrad[0:3*numT]
            elif step == 2:
                rxyzs_ks = pIdx["rXYZs"]
                aggGrad[rxyzs_ks:rxyzs_ks+3*numT] += pixelGrad[supp_ks:supp_ks+3*numT] + teethPoseGrad[3*numT:6*numT]
            elif step == 3:
                scales_ks = pIdx["scales"]
                aggGrad[scales_ks:scales_ks+numT] += pixelGrad[supp_ks:supp_ks+numT] + teethPoseGrad[6*numT:7*numT]
            elif step == 4:
                txyzs_ks = pIdx["tXYZs"]
                rxyzs_ks = pIdx["rXYZs"]
                scales_ks = pIdx["scales"]
                aggGrad[txyzs_ks:txyzs_ks+3*numT] += pixelGrad[supp_ks:supp_ks+3*numT] + teethPoseGrad[0:3*numT]
                aggGrad[rxyzs_ks:rxyzs_ks+3*numT] += pixelGrad[supp_ks+3*numT:supp_ks+6*numT] + teethPoseGrad[3*numT:6*numT]
                aggGrad[scales_ks:scales_ks+numT] += pixelGrad[supp_ks+6*numT:supp_ks+7*numT] + teethPoseGrad[6*numT:7*numT]
        elif stage == 3:
            fVec_ks = pIdx["featureVec"]
            _m = self.numTooth*self.numPC
            aggGrad[fVec_ks:fVec_ks+_m] += pixelGrad[supp_ks:supp_ks+_m] + featureVecGrad
        return aggGrad


    def maximization_step_5Views(self, stage, step, maxiter=10, verbose=True):
        x0, pIdx = self.getCurrentGlobalParamsOf5Views_as_x0(stage, step)
        if stage == 3:
            for phType in self.photoTypes:
                self.SigmaT_segs[phType.value] = self.updateCorreSigmaTSegs(phType)

        # param bounds
        bounds = self.getParamBounds(x0, pIdx, stage, step)
        optRes = scipy.optimize.minimize(fun=self.MStepLoss, x0=x0, jac=True, bounds=bounds, args=(pIdx, stage, step, False, True), \
            method="SLSQP", tol=1e-6, options={"ftol":1e-6,"maxiter":maxiter,"disp":False})
        params = optRes.x

        # update params
        self.ex_rxyz, self.ex_txyz, self.focLth, self.dpix, self.u0, self.v0, self.rela_rxyz, self.rela_txyz =\
            self.parseGlobalParamsOf5Views(params, pIdx)
        if stage == 1:
            self.rowScaleXZ = params[pIdx["rowScaleXZ"]:pIdx["rowScaleXZ"]+2]
        elif stage == 2:
            self.transVecXYZs, self.rotVecXYZs, self.scales = self.parseTeethPoseParams(params, pIdx, step)
        elif stage == 3:
            self.featureVec = params[pIdx["featureVec"]:pIdx["featureVec"]+self.numTooth*self.numPC].reshape(self.featureVec.shape)
        self.loss_maximization_step = self.MStepLoss(params, pIdx, stage, step, verbose, return_grad=False)


    def getParamBounds(self, x0, pIdx, stage, step):
        """Get bounds of params"""
        bounds = []
        ex_rxyz_d = 0.3
        ex_txyz_d = 20.
        ex_rxyz_params = x0[pIdx["ex_rxyz"]:pIdx["ex_rxyz"]+15]
        ex_txyz_params = x0[pIdx["ex_txyz"]:pIdx["ex_txyz"]+15]
        rela_rxyz_d = 0.05
        rela_txyz_d = 1.
        rela_rxyz_params = x0[pIdx["rela_rxyz"]:pIdx["rela_rxyz"]+3]
        rela_txyz_params = x0[pIdx["rela_txyz"]:pIdx["rela_txyz"]+3]
        
        ex_rxyz_bounds = np.stack([ex_rxyz_params-ex_rxyz_d, ex_rxyz_params+ex_rxyz_d])
        ex_rxyz_bounds = list(zip(ex_rxyz_bounds[0], ex_rxyz_bounds[1])) # list of tuples
        ex_txyz_bounds = np.stack([ex_txyz_params-ex_txyz_d, ex_txyz_params+ex_txyz_d])
        ex_txyz_bounds = list(zip(ex_txyz_bounds[0], ex_txyz_bounds[1])) # list of tuples

        focLth_bounds = [(30., 150.)] * 5
        dpix_bounds = [(None,None)] * 5
        u0_bounds = [(300., 500.)] * 5
        v0_bounds = [(200., 400.)] * 5
        intr_bounds = focLth_bounds + dpix_bounds + u0_bounds + v0_bounds

        rela_rxyz_bounds = [(-rela_rxyz_d, rela_rxyz_d)] * 3
        rela_txyz_bounds = np.stack([rela_txyz_params-rela_txyz_d, rela_txyz_params+rela_txyz_d])
        rela_txyz_bounds = list(zip(rela_txyz_bounds[0], rela_txyz_bounds[1])) # list of tuples
        bounds = ex_rxyz_bounds + ex_txyz_bounds + intr_bounds + rela_rxyz_bounds + rela_txyz_bounds
        if stage == 1:
            bounds = bounds + [(0.01, 2.)] * 2 # add bounds for rowScaleXZ
        elif stage == 2:
            numT = self.numTooth
            tXYZs_d = 10.*self.transVecStd
            rXYZs_d = 4.*self.rotVecStd
            scales_d = 4.*self.scaleStd
            if step == 1:
                bounds += [(-tXYZs_d,tXYZs_d)] * (3*numT) # add bounds for tooth translation vecs
            elif step == 2:
                bounds += [(-rXYZs_d,rXYZs_d)] * (3*numT) # add bounds for tooth rot vecs
            elif step == 3:
                bounds += [(1.-scales_d,1.+scales_d)] * numT # add bounds for tooth scales
            elif step == 4:
                bounds += [(-tXYZs_d,tXYZs_d)] * (3*numT) + [(-rXYZs_d,rXYZs_d)] * (3*numT) + [(1.-scales_d,1.+scales_d)] * numT
        elif stage == 3:
            bounds += [(-5.,5.)] * (self.numTooth*self.numPC) # add bounds for featureVec (mean=0,std=1)
        return bounds

    
    ###########################################
    ######### Maximization stage 3 ############
    ###########################################

    
    def updateCorreSigmaTSegs(self, photoType):
        ph = photoType.value
        tIdx = self.visIdx[ph]
        SigmaT_segs = []
        for sigmaT,vis_hull_vids in zip(self.SigmaT[tIdx], self.vis_hull_vertex_indices[ph]): # self.SigmaT.shape=(numTooth,numPC,numPoint*3)
            sigmaTseg = sigmaT.reshape(self.numPC, self.numPoint, 3)[:,vis_hull_vids,:]
            SigmaT_segs.append(sigmaTseg.reshape(self.numPC, 3*len(vis_hull_vids)))
        return SigmaT_segs


    ###################################
    ######### Save H5 File ############
    ###################################
    
    def saveDemo2H5(self, h5File, patientId, X_Ref):
        if not os.path.exists(os.path.dirname(h5File)):
            os.makedirs(os.path.dirname(h5File))
        with h5py.File(h5File,'w') as f: #每次覆盖写入
            grp = f.create_group(str(patientId))
            grp.create_dataset("UPPER_INIT", data=np.array(self.X_Mu[:self.numUpperTooth], dtype=np.double))
            grp.create_dataset("LOWER_INIT", data=np.array(self.X_Mu[self.numUpperTooth:], dtype=np.double))
            grp.create_dataset("UPPER_PRED", data=np.array(self.X_trans[:self.numUpperTooth], dtype=np.double))
            grp.create_dataset("LOWER_PRED", data=np.array(self.X_trans[self.numUpperTooth:], dtype=np.double))
            grp.create_dataset("UPPER_REF", data=np.array(X_Ref[:self.numUpperTooth], dtype=np.double))
            grp.create_dataset("LOWER_REF", data=np.array(X_Ref[self.numUpperTooth:], dtype=np.double))
            grp.create_dataset("MASK", data=np.array(self.Mask, dtype=np.double))
            grp.create_dataset("RELA_R", data=np.array(self.rela_R, dtype=np.double))
            grp.create_dataset("RELA_T", data=np.array(self.rela_txyz, dtype=np.double))
            grp.create_dataset("EX_RXYZ", data=np.array(self.ex_rxyz, dtype=np.double))
            grp.create_dataset("EX_TXYZ", data=np.array(self.ex_txyz, dtype=np.double))
            grp.create_dataset("FOCLTH", data=np.array(self.focLth, dtype=np.double))
            grp.create_dataset("DPIX", data=np.array(self.dpix, dtype=np.double))
            grp.create_dataset("U0", data=np.array(self.u0, dtype=np.double))
            grp.create_dataset("V0", data=np.array(self.v0, dtype=np.double))
            grp.create_dataset("SCALES", data=np.array(self.scales, dtype=np.double))
            grp.create_dataset("ROT_VEC_XYZS", data=np.array(self.rotVecXYZs, dtype=np.double))
            grp.create_dataset("TRANS_VEC_XYZS", data=np.array(self.transVecXYZs, dtype=np.double))
            grp.create_dataset("FEATURE_VEC", data=np.array(self.featureVec, dtype=np.double))


