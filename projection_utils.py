import glob
import os
import numpy as np
import enum
import pandas as pd
import open3d as o3d
from matplotlib import pyplot as plt
import scipy.io
import utils
import skimage
from scipy.spatial.transform import Rotation as RR
import h5py
from utils import LOWER_INDICES, UPPER_INDICES



# 定义每张图片中采用的牙齿轮廓
MASK_UPPER = np.array([ True,  True,  True,  True,  True,  True,  True,  # 上牙列视图
                  True,  True,  True,  True,  True,  True,  True,  
                  False,  False,  False,  False,  False,  False,  False, 
                  False,  False,  False,  False,  False,  False,  False])

MASK_LOWER = np.array([False,  False,  False,  False,  False,  False,  False, # 下牙列视图
                  False,  False,  False,  False,  False,  False,  False, 
                  True,  True,  True,  True,  True,  True,  True,  
                  True,  True,  True,  True,  True,  True,  True])

MASK_LEFT = np.array([ True,  False,  False,  False,  False,  False,  False,  # 左视图
                  True,  True,  True,  True,  True,  True,  False,  
                  True,  True,  True,  True,  True,  True,  False, 
                  True,  False,  False,  False,  False,  False,  False])

MASK_RIGHT = np.array([ True,  True,  True,  True,  True,  True,  False,  # 右视图
                  True,  False,  False,  False,  False,  False,  False,  
                  True,  False,  False,  False,  False,  False,  False, 
                  True,  True,  True,  True,  True,  True,  False])

MASK_FRONTAL = np.array([ True,  True,  True,  True,  False,  False,  False,  # 正视图
                  True,  True,  True,  True,  False,  False,  False,  
                  True,  True,  True,  True,  False,  False,  False, 
                  True,  True,  True,  True,  False,  False,  False])


@enum.unique
class PHOTO(enum.Enum):
    # Enum values must be 0,1,2,3,4 
    UPPER = 0
    LOWER = 1
    LEFT = 2
    RIGHT = 3
    FRONTAL = 4




#####################################################
######## related to Edge Mask & photos ##############
#####################################################



def getEdgeMask(edge_mask_path, name_idx_df, TagID, photo_types, resized_width=800, binary=True, activation_thre=0.1):
    # 获取指定TagID的病人的五张二值牙齿边界图片
    photoFiles = name_idx_df.loc[name_idx_df["index"]==TagID, photo_types].values[0]
    edgeMaskFiles = [os.path.join(edge_mask_path, os.path.splitext(os.path.basename(pf))[0]+".png") for pf in photoFiles]
    print(edgeMaskFiles)
    edgeMasks = []
    for edgeMaskFile in edgeMaskFiles:
        edgeMask = skimage.io.imread(edgeMaskFile, as_gray=True)
        rescale = resized_width / edgeMask.shape[1]
        edgeMask = skimage.transform.rescale(edgeMask, rescale, anti_aliasing=True)
        edgeMask[edgeMask<activation_thre] = 0.
        if binary == True:
            thre = 0.5
            edgeMask[edgeMask<thre] = 0.
            edgeMask[edgeMask>=thre] = 1.
        # edgeMask = skimage.morphology.binary_erosion(edgeMask, footprint=skimage.morphology.disk(2))
        edgeMasks.append(edgeMask)
    return edgeMasks

def getPhotos(photo_path, name_idx_df, TagID, photo_types, new_shape=(1080,1440)):
    # 获取指定TagID的病人的五张二值牙齿边界图片
    photoFiles = name_idx_df.loc[name_idx_df["index"]==TagID, photo_types].values[0]
    photoFiles = [os.path.join(photo_path, os.path.splitext(os.path.basename(pf))[0]+".png") for pf in photoFiles]
    print(photoFiles)
    photos = []
    for photoFile in photoFiles:
        photo = skimage.io.imread(photoFile, as_gray=False)
        photo = skimage.transform.resize(photo, new_shape, anti_aliasing=True)
        photos.append(photo)
    return photos

def getRgbEdgeMask(edge_mask_path, name_idx_df, TagID, photo_types, resized_width=800):
    # 获取指定TagID的病人的五张RGB牙齿边界图片
    photoFiles = name_idx_df.loc[name_idx_df["index"]==TagID, photo_types].values[0]
    edgeMaskFiles = [os.path.join(edge_mask_path, os.path.splitext(os.path.basename(pf))[0]+".png") for pf in photoFiles]
    print(edgeMaskFiles)
    edgeMasks = []
    for edgeMaskFile in edgeMaskFiles:
        edgeMask = skimage.io.imread(edgeMaskFile, as_gray=False)
        rescale = resized_width / edgeMask.shape[1]
        edgeMask = skimage.transform.rescale(edgeMask, rescale, anti_aliasing=True, channel_axis=2)
        thre = 0.9
        edgeMask[edgeMask<thre] = 0.
        edgeMask[edgeMask>=thre] = 1.
        # edgeMask = skimage.morphology.binary_erosion(edgeMask, footprint=skimage.morphology.disk(2))
        edgeMasks.append(edgeMask[...,:3])
    return edgeMasks

def visualizeEdgeMasks(edgeMasks, photo_types, cmap='gray'):
    plt.figure(figsize=(16,16))
    for i,ptype in enumerate(photo_types):
        plt.subplot(3,2,i+1)
        plt.imshow(edgeMasks[i], cmap=cmap)
        plt.title(ptype)
        print("num of contour points in {}: {} ".format(ptype, np.sum(edgeMasks[i]>0)))


#####################################################
############ related to pointcloud ##################
#####################################################


def GetPGByTagId(pg_npy, TagId):
    # PGs.shape = (?,28,1500,3)
    PGs = np.load(pg_npy)
    return PGs[TagId]

def GetMaskByTagId(mask_npy, TagId):
    # Masks.shape = (?,28)
    Masks = np.load(mask_npy)
    return np.squeeze(Masks[TagId]) # shape=(28,)

def __getToothIndex(f):
    return int(os.path.basename(f).split(".")[0].split("_")[-1])

def loadMuEigValSigma(ssmDir, numPC):
    """Mu.shape=(28,1500,3), sqrtEigVals.shape=(28,1,100), Sigma.shape=(28,4500,100)"""
    muNpys = glob.glob(os.path.join(ssmDir,"meanAlignedPG_*.npy"))
    muNpys = sorted(muNpys, key=lambda x:__getToothIndex(x))
    Mu = np.array([np.load(x) for x in muNpys])
    eigValNpys = glob.glob(os.path.join(ssmDir,"eigVal_*.npy"))
    eigValNpys = sorted(eigValNpys, key=lambda x:__getToothIndex(x))
    sqrtEigVals = np.sqrt(np.array([np.load(x) for x in eigValNpys]))
    eigVecNpys = glob.glob(os.path.join(ssmDir,"eigVec_*.npy"))
    eigVecNpys = sorted(eigVecNpys, key=lambda x:__getToothIndex(x))
    Sigma = np.array([np.load(x) for x in eigVecNpys])
    return Mu, sqrtEigVals[:,np.newaxis,:numPC], Sigma[...,:numPC]

def GetPgRefUL(PgRef, Mask):
    # 获得病人上下牙列的三维金标准
    PG_U, PG_L = np.split(PgRef, 2, axis=0)
    Mask_U, Mask_L = np.split(Mask, 2, axis=0)
    return PG_U[Mask_U], PG_L[Mask_L]


#####################################################
######## related to Registration Params #############
#####################################################


def loadInvRegistrationParams(loadDir):
    """生成DataFrame,含有逆配准参数s,R,transVec,transVecShift"""
    """initPG = np.multiply(s, np.matmul(PG+transVec, R)) + transVecShift"""
    toothIndices = UPPER_INDICES + LOWER_INDICES
    paramDF = pd.DataFrame(columns=["tag"])
    # 下牙列逆配准参数
    tags, rowScales, transVecShifts = utils.readToothRowScalesFromHDF5(os.path.join(loadDir, "scalesOfLowerToothRow.hdf5"), "L")
    indexTag = [int(tag[:-1]) for tag in tags]
    invScales = [1./s for s in rowScales]
    invTransVecShifts = -transVecShifts
    tempDF = pd.DataFrame({"tag":indexTag,"lower_s":list(invScales), "lower_ts":list(invTransVecShifts)})
    paramDF = paramDF.merge(tempDF, how="outer", on="tag")
    # 上牙列逆配准参数
    tags, rowScales, transVecShifts = utils.readToothRowScalesFromHDF5(os.path.join(loadDir, "scalesOfUpperToothRow.hdf5"), "U")
    indexTag = [int(tag[:-1]) for tag in tags]
    invScales = [1./s for s in rowScales]
    invTransVecShifts = -transVecShifts
    tempDF = pd.DataFrame({"tag":indexTag,"upper_s":list(invScales), "upper_ts":list(invTransVecShifts)})
    paramDF = paramDF.merge(tempDF, how="outer", on="tag")
    # 牙齿统计形状逆配准参数
    for i in toothIndices:
        h5File = os.path.join(loadDir, "sRtParams_{}.hdf5".format(i))
        tags, scales, rotMats, transVecs = utils.readRegistrationParamsFromHDF5(h5File, i)
        indexTag = [int(tag[:-1]) for tag in tags]
        invRotVecs = RR.from_matrix(rotMats).as_rotvec() # 两次求逆（转置）相互抵消
        invScales = 1./scales
        invTransVecs = -transVecs
        tempDF = pd.DataFrame({"tag":indexTag,"{}s".format(i):list(invScales), "{}rx".format(i):list(invRotVecs[:,0]), \
                               "{}ry".format(i):list(invRotVecs[:,1]), "{}rz".format(i):list(invRotVecs[:,2]), \
                               "{}tx".format(i):list(invTransVecs[:,0]), "{}ty".format(i):list(invTransVecs[:,1]), \
                               "{}tz".format(i):list(invTransVecs[:,2])})
        paramDF = paramDF.merge(tempDF, how="outer", on="tag")
    sUpperColumns = ["{}s".format(id) for id in UPPER_INDICES]
    sLowerColumns = ["{}s".format(id) for id in LOWER_INDICES]
    paramDF = paramDF[~paramDF[sUpperColumns].isna().all(axis=1)]
    paramDF = paramDF[~paramDF[sLowerColumns].isna().all(axis=1)] # 删除缺少上牙列或下牙列的数据
    paramDF = paramDF.sort_values(by="tag", ignore_index=True)
    return paramDF


def getScalesRxyzTxyzRotMats(invParamDF, index):
    """计算s,rxyz,txyz,rotMats (nan用0填充)
    txyz = np.multiply(s, np.matmul(transVec, rotMats)) + transVecShift"""
    toothIndices = UPPER_INDICES + LOWER_INDICES
    numTooth = len(toothIndices)
    sColumns = ["{}s".format(id) for id in toothIndices]
    rxyzColumns = ["{}r{}".format(id, p) for id in toothIndices for p in ["x","y","z"]]
    postTxyzColumns = ["{}t{}".format(id, p) for id in toothIndices for p in ["x","y","z"]]
    indexedRow = invParamDF[invParamDF["tag"]==index]
    
    scales = indexedRow[sColumns].values.flatten().tolist()
    scales = np.nan_to_num(scales, nan=0.0)
    scales = np.array(scales)[:,np.newaxis,np.newaxis]
    
    rxyzs = indexedRow[rxyzColumns].to_numpy().reshape(numTooth, 3)
    rxyzs = np.nan_to_num(rxyzs, nan=0.0)
    rotMats = RR.from_rotvec(rxyzs).as_matrix()
    rotMats = np.transpose(rotMats, (0,2,1))
    
    postTxyzs = indexedRow[postTxyzColumns].to_numpy().reshape(numTooth, 3)
    transVecs = np.nan_to_num(postTxyzs, nan=0.0)
    transVecShifts = np.vstack([indexedRow["upper_ts"].values[0],indexedRow["lower_ts"].values[0]])
    transVecShifts = np.nan_to_num(transVecShifts, nan=0.0)
    txyzs = np.multiply(scales, np.matmul(np.array(transVecs)[:,np.newaxis,:], np.array(rotMats))) + np.array(transVecShifts)[:,np.newaxis,:]
    
    return np.array(scales), np.array(rxyzs), np.array(txyzs), np.array(rotMats)

def loadAlignedTeethPgWithMaskByIndex(index, pgShape, srcRootDir):
    pgTxts = [os.path.join(srcRootDir, str(tID), "{}U.txt".format(index)) for tID in UPPER_INDICES]
    pgTxts.extend([os.path.join(srcRootDir, str(tID), "{}L.txt".format(index)) for tID in LOWER_INDICES])
    masks = [os.path.exists(f) for f in pgTxts]
    PGs = [np.loadtxt(f) if mask==True else np.zeros(pgShape) for f,mask in zip(pgTxts,masks)]
    return np.array(masks), np.array(PGs)

def loadDataSet(invParamDF, Mu, Sigma, pgShape, srcRootDir):
    Y_mask = [] # shape=(None,28,1,1)
    Y_scale = [] # shape=(None,28,1,1)
    Y_rxyz = [] # shape=(None,28,3)
    Y_txyz = [] # shape=(None,28,1,3)
    Y_fVec = [] # shape=(None,28,1,NUM_PC)
    X_pg = [] # shape=(None,28,1500,3)
    for i,index in enumerate(invParamDF["tag"].values):
        mask, alignedPG = loadAlignedTeethPgWithMaskByIndex(index, pgShape, srcRootDir)
        mask = mask.reshape(-1,1,1)
        fVec = np.matmul((alignedPG - Mu).reshape(Mu.shape[0], 1, Mu.shape[1]*Mu.shape[2]), Sigma)
        scale, rxyz, txyz, rotMat = getScalesRxyzTxyzRotMats(invParamDF, index)
        PG = np.multiply(mask, np.multiply(np.matmul(alignedPG, rotMat), scale) + txyz)
        Y_mask.append(mask)
        Y_scale.append(scale)
        Y_rxyz.append(rxyz)
        Y_txyz.append(txyz)
        Y_fVec.append(fVec)
        X_pg.append(PG)
        print("Load {}/{}".format(i+1,len(invParamDF["tag"].values)))
    return X_pg, Y_mask, Y_scale, Y_rxyz, Y_txyz, Y_fVec


def updateAbsTransVecs(invParamDF, Mu):
    """将牙列transVecShift融入txyz中，忽略牙列scale的影响，并且将txyz定义在局部坐标系下"""
    toothIndices = UPPER_INDICES+LOWER_INDICES
    invParamCopy = invParamDF.copy()
    numSample = invParamDF.shape[0]
    numTooth = len(toothIndices)
    X_Mu_centroids = {tID:Mu[i].mean(axis=0) for i,tID in enumerate(toothIndices)}
    invScalesColumns = ["{}s".format(id) for id in toothIndices]
    invRotAngleXYZColumns = ["{}r{}".format(id, p) for id in toothIndices for p in ["x","y","z"]]
    invTransVecXYZColumns = ["{}t{}".format(id, p) for id in toothIndices for p in ["x","y","z"]]
    invTransVecShiftColumns = ["upper_ts", "lower_ts"]
    invScales = invParamDF[invScalesColumns].to_numpy()
    invRotVecs = invParamDF[invRotAngleXYZColumns].to_numpy().reshape(numSample, numTooth, 3)
    invTransVecs = invParamDF[invTransVecXYZColumns].to_numpy().reshape(numSample, numTooth, 3)
    invTransVecShifts = np.concatenate([np.stack(invParamDF["upper_ts"].to_list()), np.stack(invParamDF["lower_ts"].to_list())], axis=1)
    
    for i in range(numSample):
        for j,tID in enumerate(toothIndices):
            rxyz = invRotVecs[i,j]
            invRotMat = RR.from_rotvec(rxyz).as_matrix().T
            invTx, invTy, invTz = - X_Mu_centroids[tID] + invScales[i,j] * (invTransVecs[i,j] + X_Mu_centroids[tID] - invTransVecShifts[i,j]) @ invRotMat
            invParamCopy.loc[i,"{}tx".format(tID)] = invTx
            invParamCopy.loc[i,"{}ty".format(tID)] = invTy
            invParamCopy.loc[i,"{}tz".format(tID)] = invTz
    invParamCopy = invParamCopy.drop(labels=["upper_s", "lower_s", "upper_ts", "lower_ts"], axis=1)
    return invParamCopy
            
            

def getMeanAndVarianceOfInvRegistrationParams(invParamDF):
    toothIndices = UPPER_INDICES+LOWER_INDICES
    numTooth = len(toothIndices)
    invScalesColumns = ["{}s".format(id) for id in toothIndices]
    invRotAngleXYZColumns = ["{}r{}".format(id, p) for id in toothIndices for p in ["x","y","z"]]
    invTransVecXYZColumns = ["{}t{}".format(id, p) for id in toothIndices for p in ["x","y","z"]]
    invScales = invParamDF[invScalesColumns].to_numpy()
    invScaleMeans = np.nanmean(invScales, axis=0)
    invScaleVars = np.nanvar(invScales, ddof=1, axis=0)
    invRotAngleXYZs = invParamDF[invRotAngleXYZColumns].to_numpy()
    invRotAngleXYZMeans = np.nanmean(invRotAngleXYZs, axis=0).reshape(numTooth,3)
    invRotAngleXYZVars = np.nanvar(invRotAngleXYZs, ddof=1, axis=0).reshape(numTooth,3)
    invTransVecXYZs = invParamDF[invTransVecXYZColumns].to_numpy()
    invTransVecXYZMeans = np.nanmean(invTransVecXYZs, axis=0).reshape(numTooth,3)
    invTransVecXYZVars = np.nanvar(invTransVecXYZs, ddof=1, axis=0).reshape(numTooth,3)
    return invScaleMeans, invScaleVars, invRotAngleXYZMeans, invRotAngleXYZVars, invTransVecXYZMeans, invTransVecXYZVars


def GetPoseCovMats(invParamDF, toothIndices):
    # 每个位置的牙齿的6个变换参数的协方差矩阵,shape=(28,6,6)
    suffixes = ["tx","ty","tz","rx","ry","rz"]
    covMats = []
    for id in toothIndices:
        cols = [str(id)+suffix for suffix in suffixes]
        A = invParamDF[cols].to_numpy()
        covA = np.ma.cov(np.ma.masked_invalid(A), rowvar=False)
        assert not covA.mask.any() #检查是否有nan
        covMat = covA.data
        assert utils.is_pos_def(covMat)
        # variances = np.ma.var(np.ma.masked_invalid(A), axis=0, ddof=1)
        # assert not variances.mask.any() #检查是否有nan
        # std = np.sqrt(variances.data)
        # rhoCoef = covMat / np.multiply(std[:,None],std)
        # print(rhoCoef)
        covMats.append(covMat)
    return np.array(covMats)

def GetScaleCovMat(invParamDF, toothIndices):
    # 牙齿scale的协方差矩阵,shape=(28,28)
    cols = [str(i)+"s" for i in toothIndices]
    A = invParamDF[cols]
    covA = np.ma.cov(np.ma.masked_invalid(A), rowvar=False)
    assert not covA.mask.any() #检查是否有nan
    covMat = covA.data
    assert utils.is_pos_def(covMat)
    return covMat






###############################
############ Demo #############
###############################


def saveTempEmOptParamsWithXRef(matFileName, emopt, X_Ref):
    scipy.io.savemat(matFileName, {"np_invCovMatOfPose":emopt.invCovMats, "np_invCovMatOfScale":emopt.invCovMatOfScale,
                    "np_ex_rxyz":emopt.ex_rxyz, "np_ex_txyz":emopt.ex_txyz, "np_focLth":emopt.focLth, "np_dpix":emopt.dpix, 
                    "np_u0":emopt.u0, "np_v0":emopt.v0, "np_rela_rxyz":emopt.rela_rxyz, "np_rela_txyz":emopt.rela_txyz, "np_rowScaleXZ":emopt.rowScaleXZ, 
                    "np_scales":emopt.scales, "np_rotAngleXYZs":emopt.rotAngleXYZs, "np_transVecXYZs":emopt.transVecXYZs,
                    "np_X_Mu":emopt.X_Mu, "np_X_Mu_pred":emopt.X_Mu_pred, "np_X_Mu_pred_normals":emopt.X_Mu_pred_normals,
                    "np_visIdx":emopt.visIdx, "np_corre_pred_idx":emopt.corre_pred_idx, "np_P_true":emopt.P_true_95_percentile, "np_X_ref":X_Ref})



def showPredPointClouds(y, py, mode=0):
    def __rotate_view(vis):
        ctr = vis.get_view_control()
        if mode == 1:
            ctr.rotate(4.0, 0.0)
        elif mode == 2:
            ctr.rotate(0.0, 4.0)
        elif mode == 3:
            ctr.rotate(0.0, -4.0)
        return False

    pcdY = o3d.geometry.PointCloud()
    pcdY.points = o3d.utility.Vector3dVector(y)
    pcdY.paint_uniform_color(np.array([1.,0.,0.]))
    pcdPY = o3d.geometry.PointCloud()
    pcdPY.points = o3d.utility.Vector3dVector(py)
    pcdPY.paint_uniform_color(np.array([0.,0.,1.]))
    if mode == 0: # No rotation animation
        o3d.visualization.draw_geometries([pcdY,pcdPY], window_name="Ground Truth PCL(red) and Prediction PCL(blue)", width=800, height=600, left=50,top=50, point_show_normal=False)
    elif mode in [1,2,3]: # rotate
        o3d.visualization.draw_geometries_with_animation_callback([pcdY,pcdPY], __rotate_view, window_name="Ground Truth PCL(red) and Prediction PCL(blue)", width=800, height=600, left=50,top=50)
    else:
        print("Invalid mode input; expected 0,1,2,3.")


def saveDemo2H5(h5File, emopt, patientId, X_Ref):
    if not os.path.exists(os.path.dirname(h5File)):
        os.makedirs(os.path.dirname(h5File))
    with h5py.File(h5File,'w') as f: #每次覆盖写入
        grp = f.create_group(str(patientId))
        grp.create_dataset("UPPER_INIT", data=np.array(emopt.X_Mu[:emopt.numUpperTooth], dtype=np.double))
        grp.create_dataset("LOWER_INIT", data=np.array(emopt.X_Mu[emopt.numUpperTooth:], dtype=np.double))
        grp.create_dataset("UPPER_PRED", data=np.array(emopt.X_trans[:emopt.numUpperTooth], dtype=np.double))
        grp.create_dataset("LOWER_PRED", data=np.array(emopt.X_trans[emopt.numUpperTooth:], dtype=np.double))
        grp.create_dataset("UPPER_REF", data=np.array(X_Ref[:emopt.numUpperTooth], dtype=np.double))
        grp.create_dataset("LOWER_REF", data=np.array(X_Ref[emopt.numUpperTooth:], dtype=np.double))
        grp.create_dataset("MASK", data=np.array(emopt.Mask, dtype=np.double))
        grp.create_dataset("RELA_R", data=np.array(emopt.rela_R, dtype=np.double))
        grp.create_dataset("RELA_T", data=np.array(emopt.rela_txyz, dtype=np.double))
        grp.create_dataset("EX_RXYZ", data=np.array(emopt.ex_rxyz, dtype=np.double))
        grp.create_dataset("EX_TXYZ", data=np.array(emopt.ex_txyz, dtype=np.double))
        grp.create_dataset("FOCLTH", data=np.array(emopt.focLth, dtype=np.double))
        grp.create_dataset("DPIX", data=np.array(emopt.dpix, dtype=np.double))
        grp.create_dataset("U0", data=np.array(emopt.u0, dtype=np.double))
        grp.create_dataset("V0", data=np.array(emopt.v0, dtype=np.double))
        grp.create_dataset("SCALES", data=np.array(emopt.scales, dtype=np.double))
        grp.create_dataset("ROT_ANGLE_XYZS", data=np.array(emopt.rotAngleXYZs, dtype=np.double))
        grp.create_dataset("TRANS_VEC_XYZS", data=np.array(emopt.transVecXYZs, dtype=np.double))
        grp.create_dataset("FEATURE_VEC", data=np.array(emopt.featureVec, dtype=np.double))
        
def readDemoFromH5(h5File, patientId):
    with h5py.File(h5File, 'r') as f:
        grp = f[str(patientId)]
        X_Mu_Upper = grp["UPPER_INIT"][:]
        X_Mu_Lower = grp["LOWER_INIT"][:]
        X_Pred_Upper = grp["UPPER_PRED"][:]
        X_Pred_Lower = grp["LOWER_PRED"][:]
        X_Ref_Upper = grp["UPPER_REF"][:]
        X_Ref_Lower = grp["LOWER_REF"][:]
        rela_R = grp["RELA_R"][:]
        rela_t = grp["RELA_T"][:]
    return X_Mu_Upper, X_Mu_Lower, X_Pred_Upper, X_Pred_Lower, X_Ref_Upper, X_Ref_Lower, rela_R, rela_t

def readCameraParamsFromH5(h5File, patientId):
    with h5py.File(h5File, 'r') as f:
        grp = f[str(patientId)]
        ex_rxyz = grp["EX_RXYZ"][:]
        ex_txyz = grp["EX_TXYZ"][:]
        focLth = grp["FOCLTH"][:]
        dpix = grp["DPIX"][:]
        u0 = grp["U0"][:]
        v0 = grp["V0"][:]
        rela_R = grp["RELA_R"][:]
        rela_t = grp["RELA_T"][:]
        return ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_R, rela_t