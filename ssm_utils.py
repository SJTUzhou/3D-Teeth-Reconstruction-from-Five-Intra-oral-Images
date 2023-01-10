import open3d as o3d
import numpy as np
import shutil
import os
import h5py
import glob
import matplotlib.pyplot as plt


UPPER_INDICES = [11,12,13,14,15,16,17,21,22,23,24,25,26,27] 
LOWER_INDICES = [31,32,33,34,35,36,37,41,42,43,44,45,46,47] 


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity, ord="fro")
    detR = np.linalg.det(R)
    return (n < 1e-6) and (np.abs(detR-1) < 1e-6)


def rotationMatrixToEulerAngles(RotMat):
    """Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped )."""
    R = RotMat.T # https://blog.csdn.net/weixin_39675633/article/details/103434557
    assert(isRotationMatrix(R))
    cosy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    siny = -R[2,0]
    singular = cosy < 1e-6
    x, y, z = (0., 0., 0.)
    if not singular:
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], cosy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        if np.abs(siny-1) < 1e-6:
            z = 0
            x = z + np.arctan2(R[0,1], R[0,2])
            y = np.pi / 2.
        else:
            z = 0
            x = -z + np.arctan2(-R[0,1], -R[0,2])
            y = -np.pi / 2.
    return np.array([x, y, z])


def getRotAngles(RotMats):
    return np.vstack([rotationMatrixToEulerAngles(R) for R in RotMats])

def getRotMat(rxryrz):
    return o3d.geometry.get_rotation_matrix_from_zyx(rxryrz[::-1]).T #先进行x轴旋转，再y轴，再z轴；取转置表示右乘旋转矩阵

def getRotMats(rotAnglesXYZ):
    # rotMats = []
    # for rxryrz in rotAnglesXYZ:
    #     rotMat = getRotMat(rxryrz) 
    #     rotMats.append(rotMat)
    # return np.array(rotMats)
    return np.apply_along_axis(getRotMat, axis=1, arr=rotAnglesXYZ)



def saveAlignedPointGroups2Txt(alignedPointGroups, trainPgTags, saveDir):
    if os.path.exists(saveDir):
        shutil.rmtree(saveDir)
    os.makedirs(saveDir)
    for i,pgID in enumerate(trainPgTags):
        np.savetxt(os.path.join(saveDir,"{}.txt".format(pgID)), alignedPointGroups[i])
        
def loadAlignedPointGroupsWithIndex(loadDir):
    txtFiles = glob.glob(os.path.join(loadDir, "*.txt"))
    txtFileIndices = [os.path.basename(f).split('.')[0] for f in txtFiles]
    alignedPointGroups = [np.loadtxt(f) for f in txtFiles]
    return alignedPointGroups, txtFileIndices

def remainedInfoRatio(eigVal, numPC):
    if len(eigVal)<numPC:
        return 1.
    else:
        return np.sum(eigVal[:numPC])/np.sum(eigVal)


def getEigValVecOfSSMByPCA(alignedPointGroups): 
    alignedPointVectors = np.array([pg.flatten() for pg in alignedPointGroups], dtype=np.float32) # shape=(sampleNum, 3*pointNum)
    meanTrainPointVector = alignedPointVectors.mean(axis=0)
    normalizedTrainPointVector = alignedPointVectors - meanTrainPointVector #均值为0标准化
    A = normalizedTrainPointVector
    sampleNum = A.shape[0]
    convMat = 1./(sampleNum-1) * A.T @ A #需要对协方差矩阵A.T @ A进行主成分分析
    equConvMat = 1./(sampleNum-1) * A @ A.T # A@A.T和A.T@A的特征值相同
    eigVal, equEigVec = np.linalg.eig(equConvMat) # equEigVec是A @ A.T的特征值
    eigVal[eigVal<1e-4] = 0.0 #避免计算误差导致特征值出现负数
    eigVec = A.T @ equEigVec 
    eigVec = eigVec / np.linalg.norm(eigVec, axis=0) # 单位化 eigVec是A.T @ A的特征值
    eigOrder = sorted(range(len(eigVal)), key=lambda x:eigVal[x], reverse=True) # eigVal从大到小的索引排序
    eigVal = eigVal[eigOrder]
    eigVec = eigVec[:,eigOrder]
    return eigVal, eigVec, A, meanTrainPointVector

def saveEigValVec(srcRootDir, NumPC2Save):
    toothIndices = UPPER_INDICES + LOWER_INDICES
    for toothIndex in toothIndices:
        srcDir = os.path.join(srcRootDir, str(toothIndex))
        alignedPointGroups, alignedPgTags = loadAlignedPointGroupsWithIndex(srcDir)
        eigVal, eigVec, A, meanTrainPointVector = getEigValVecOfSSMByPCA(alignedPointGroups)
        meanAlignedPG = meanTrainPointVector.reshape(-1,3)
        eigVal = eigVal[:NumPC2Save]
        eigVec = eigVec[:,:NumPC2Save]
        saveDir = os.path.join(srcRootDir, "eigValVec")
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        np.save(os.path.join(saveDir,"eigVal_{}.npy".format(toothIndex)), eigVal)
        np.save(os.path.join(saveDir,"eigVec_{}.npy".format(toothIndex)), eigVec)
        np.save(os.path.join(saveDir,"meanAlignedPG_{}.npy".format(toothIndex)), meanAlignedPG)



def saveRegistrationParams2Hdf5(h5File, toothIndex, sRtParams, tags):
    if not os.path.exists(os.path.dirname(h5File)):
        os.makedirs(os.path.dirname(h5File))
    encodedTags = [tag.encode() for tag in tags]
    scales = [param[0] for param in sRtParams]
    rotMats = [param[1] for param in sRtParams]
    transVecs = [param[2] for param in sRtParams]
    with h5py.File(h5File,'w') as f: #每次覆盖写入
        grp = f.create_group(str(toothIndex))
        grp.create_dataset("tag", data=encodedTags)
        grp.create_dataset("s", data=np.array(scales, dtype=np.double))
        grp.create_dataset("R", data=np.array(rotMats, dtype=np.double))
        grp.create_dataset("t", data=np.array(transVecs, dtype=np.double))



def readRegistrationParamsFromHDF5(h5File, toothIndex):
    with h5py.File(h5File, 'r') as f:
        grp = f[str(toothIndex)]
        tags = [tag.decode() for tag in grp["tag"]]
        scales = grp["s"][:]
        rotMats = grp["R"][:]
        transVecs = grp["t"][:]
    return tags, scales, rotMats, transVecs



def readToothRowScalesFromHDF5(h5File, UorL):
    with h5py.File(h5File, 'r') as f:
        grp = f["toothRow{}".format(UorL)]
        tags = [tag.decode() for tag in grp["tag"]]
        scales = grp["s"][:]
        transVecShifts = grp["ts"][:]
    return tags, scales, transVecShifts



def visualizeCompactnessOfSSM(eigVal):
    eigOrder = sorted(range(len(eigVal)), key=lambda x:eigVal[x], reverse=True) # eigVal从大到小的索引排序
    sortedEigVal = eigVal[eigOrder]
    varContri = sortedEigVal / np.sum(sortedEigVal)
    varContriCumSum = np.cumsum(varContri)
    plt.figure(figsize=(8, 6))
    plt.ylim([0.,1.])
    plt.xlim([0.,len(eigVal)])
    plt.plot(np.arange(0,len(eigVal)+1),np.hstack([np.array([0.]),varContriCumSum]),linewidth=1,color='r',linestyle='-',marker='o',markerfacecolor='r',markersize=2.5)
    plt.ylabel("Cumulative explained variance")
    plt.xlabel("Num of PCA component")
    plt.grid()
    plt.show()
    



def printPlotRegistrationParams(sRtParams, plot=True):
    scaleFactors = [param[0] for param in sRtParams]
    rotMatrices = [param[1] for param in sRtParams]
    print("mean scale:",np.array(scaleFactors).mean())
    print("max scale:",np.array(scaleFactors).max())
    print("min scale:",np.array(scaleFactors).min())
    print("Std dev of scale:",np.sqrt(np.var(np.array(scaleFactors))))
    rotAnglesXYZ = getRotAngles(rotMatrices)
    print("mean rotation XYZ angles: ",rotAnglesXYZ.mean(axis=0))
    print("max rotation XYZ angles: ",rotAnglesXYZ.max(axis=0))
    print("min rotation XYZ angles: ",rotAnglesXYZ.min(axis=0))
    print("Std dev of rotation XYZ angles: ",np.sqrt(np.var(rotAnglesXYZ, axis=0)))
    translationVectors = np.vstack([param[2] for param in sRtParams])
    print("mean translation vec:",translationVectors.mean(axis=0))
    print("max translation vec:",translationVectors.max(axis=0))
    print("min translation vec:",translationVectors.min(axis=0))
    print("Std dev of translation vec:",np.sqrt(np.var(translationVectors, axis=0)))

    if plot == True:
        bins = 30
        plt.figure(figsize=(15, 15))
        plt.subplot(331)
        plt.hist(translationVectors[:,0], bins=bins)
        plt.title("translation vector: x")
        plt.subplot(332)
        plt.hist(translationVectors[:,1], bins=bins)
        plt.title("translation vector: y")
        plt.subplot(333)
        plt.hist(translationVectors[:,2], bins=bins)
        plt.title("translation vector: z")
        plt.subplot(334)
        plt.hist(rotAnglesXYZ[:,0], bins=bins)
        plt.title("rotation angle: rx")
        plt.subplot(335)
        plt.hist(rotAnglesXYZ[:,1], bins=bins)
        plt.title("rotation angle: ry")
        plt.subplot(336)
        plt.hist(rotAnglesXYZ[:,2], bins=bins)
        plt.title("rotation angle: rz")
        plt.subplot(337)
        plt.hist(scaleFactors, bins=bins)
        plt.title("scale factors: s")
        plt.show()







