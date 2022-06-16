import cycpd
import numpy as np
import scipy
import os
import glob
import sys
import functools
import time
import ray
import psutil
from utils import LOWER_INDICES, UPPER_INDICES, farthestPointDownSample, saveAlignedPointGroups2Txt, loadAlignedPointGroupsWithIndex, getEigValVecOfSSMByPCA, \
    remainedInfoRatio, saveRegistrationParams2Hdf5, readRegistrationParamsFromHDF5, printPlotRegistrationParams
from gp_non_rigid_registration import GP_Non_Rigid_Registration



NUM_CPUS = psutil.cpu_count(logical=False) 
print = functools.partial(print, flush=True)
CONSOLE = sys.stdout

def getCorrePointPairs(X, Y):
    assert len(X) <= len(Y), "Num of point in X > Num of point in Y !"
    N = len(X)
    pointPairs = []
    dists = scipy.spatial.distance_matrix(X, Y, p=2, threshold=int(1e8))
    for i in range(N): #顺序贪婪（快）
        j = np.argmin(dists[i,:])
        pointPairs.append((i, j)) # i:ref index, j: mov index
        dists[:,j] = np.inf
    return np.array(pointPairs, dtype=np.uint32)

def extractCorreMovPoints(pMov, pointPairs):
    return pMov[pointPairs[:,1], :]

def getSortedToothPoints(toothDir):
    """toothDir: Str, 存放不同样本的同一颗牙齿的路径"""
    """按照牙齿点云的点数量从小到大对点云数组进行拍讯"""
    txtFiles = glob.glob(os.path.join(toothDir, "*.txt"))
    pointArrays = [np.loadtxt(txtF) for txtF in txtFiles]
    sortedPointArrays, sortedTxtFiles = zip(*sorted(zip(pointArrays,txtFiles), key=lambda x:x[0].shape[0]))
    return list(sortedPointArrays), list(sortedTxtFiles)

def getSortedTrainTestPointGroups(sortedToothPointGroups, sortedTags, trainTags):
    sortedTrainPGs = []
    sortedTrainTags = []
    sortedTestPGs = []
    sortedTestTags = []
    for tag, pg in zip(sortedTags, sortedToothPointGroups):
        if tag in trainTags:
            sortedTrainPGs.append(pg)
            sortedTrainTags.append(tag)
        else:
            sortedTestPGs.append(pg)
            sortedTestTags.append(tag)
    return sortedTrainPGs, sortedTrainTags, sortedTestPGs, sortedTestTags

@ray.remote
def alignPointsByRigidRegistration(pRef, pMov, step, max_step, gpReg, tolerance=1e-4, max_iter=100):
    """对两组点云缩放刚性配准，以点数较少的为参照，获取刚性对齐的具有相同数量的移动点云"""
    X = pRef
    Y = pMov
    reg = cycpd.rigid_registration(**{'X': X, 'Y': Y, 'max_iterations':max_iter,'tolerance':tolerance,'verbose':False,'print_reg_params':False})
    TY,(s,r,t) = reg.register()

    # 高斯过程变形并寻找对应点对
    gpReg.setTargetPcl(TY)
    gpReg.register(eta=0.)
    pointPairs = getCorrePointPairs(gpReg.X_deformed, TY)

    correTY = extractCorreMovPoints(TY, pointPairs)
    correY = extractCorreMovPoints(Y, pointPairs)
    if step % NUM_CPUS == 0:
        # print("unique point num: ", len(np.unique(pointPairs[:,1])))
        print("---------- Finish {}/{} ----------".format(step, max_step))
    return correY, correTY, TY, (s,r,t)

@ray.remote
def findOptimalIsoScaledRigidTransform(pRef,pMov):
    """已知pRef,pMov点的对应关系，计算最优缩放的刚性变换对应的缩放常数c,旋转矩阵R和平移向量t"""
    """pRef.shape=(M,3) pMov.shape=(M,3): R,t = argmin 1/M * sum(||Qi @ (c*R) + t - Pi||_2)"""
    assert pRef.shape == pMov.shape
    meanP = pRef.mean(axis=0)
    meanQ = pMov.mean(axis=0)
    P = pRef - meanP
    Q = pMov - meanQ
    n, dim = P.shape
    H = 1./n * Q.T @ P # shape=(3,3)
    u, s, vh = np.linalg.svd(H)
    D = np.diag([1.,1.,np.linalg.det(u @ vh)])
    R = u @ D @ vh
    varQ = np.var(Q, axis=0).sum()
    c = 1/varQ * np.sum(s) # scale factor
    t = meanP - meanQ @ (c*R)
    tpMov = pMov @ (c*R) + t
    return tpMov, (c, R, t)


def alignTestPointGroups(refPG, testPGs, tolerance=1e-4, max_iter=100):
    # 多线程执行
    print("Start aligning test point groups.")
    refPG = refPG.astype(np.double)
    remotes = []
    gpReg = GP_Non_Rigid_Registration(s=2, sigma=10, srcX=refPG, n=21)
    gpReg.compute_EigVals_EigFuncs()

    _log = sys.stdout
    sys.stdout = CONSOLE
    for i,testPG in enumerate(testPGs):
        rem = alignPointsByRigidRegistration.remote(refPG, testPG, step=i+1, gpReg=gpReg, max_step=len(testPGs), tolerance=tolerance, max_iter=max_iter)
        remotes.append(rem)
    remoteValues = ray.get(remotes) # [(correPMov, correTPMov, TPMov, param), ...]
    time.sleep(0.5) # For print function in other processes
    sys.stdout = _log
    correTestPGs = [remoteVal[0] for remoteVal in remoteValues]
    # 最后一次相似变换配准，将前一次下采样的得到的点云alignedPointGroups直接进行配准
    print("Start alignment of corresponding points")
    remoteValues = ray.get([findOptimalIsoScaledRigidTransform.remote(refPG, tempCorrePG) for tempCorrePG in correTestPGs])

    correTransformedTestPGs = [remoteVal[0] for remoteVal in remoteValues]
    params = [remoteVal[1] for remoteVal in remoteValues]
    return correTestPGs, correTransformedTestPGs, params



def getPointGroupByTag(pointGroupList, tagList, searchTag):
    for tag,pg in zip(tagList,pointGroupList):
        if tag == searchTag:
            return pg
    print("{} does not exist.".format(searchTag))
    return None



if __name__ == "__main__":
    ray.init(num_cpus=NUM_CPUS, num_gpus=1)

    
    TOOTH_INDICES = LOWER_INDICES + UPPER_INDICES
    for TOOTH_INDEX in TOOTH_INDICES:
        # TOOTH_INDEX = 11
        LOG_DIR = r"./dataWithPhoto/cpd-gp-log/"
        LogFile = os.path.join(LOG_DIR, "cpd-gp-{}.log".format(TOOTH_INDEX))
        # Log file
        # if os.path.exists(LogFile):
        #     os.remove(LogFile)
        log = open(LogFile, "a")
        sys.stdout = log

        srcDir = r"./data/cpdGpAlignedData/{}/".format(TOOTH_INDEX)
        alignedPointGroups, alignedPgTags = loadAlignedPointGroupsWithIndex(srcDir) #读取已经CPD配准的牙齿点云
        meanAlignedPG = np.array(alignedPointGroups).mean(axis=0)
        refPG = meanAlignedPG

        print("Align Tooth Index: {} in Data Set".format(TOOTH_INDEX))

        dataSetDir = r"./dataWithPhoto"
        saveDir = os.path.join(dataSetDir, "cpdGpAlignedData", str(TOOTH_INDEX))
        toothDir = os.path.join(dataSetDir, "ssa-repaired-txt", str(TOOTH_INDEX))

        sortedToothPointGroups, sortedTxtFiles = getSortedToothPoints(toothDir) #参考点云按照点云中点数量从小到大排列
        sortedFileTags = [os.path.basename(f).split('.')[0] for f in sortedTxtFiles] # 排列的文件序号
        

        sortedToothPointGroups = [farthestPointDownSample(x,3000) if x.shape[0]>3000 else x for x in sortedToothPointGroups] #对点数量较大的点云进行下采样加速,最多3000点
        print("max point num: ",sortedToothPointGroups[-1].shape)
        print("min point num: ",sortedToothPointGroups[0].shape)
        assert refPG.shape[0] <= sortedToothPointGroups[0].shape[0] #参考点云的点数量必须是最少的
        


        # ALIGNMENT PROCESS (多线程)
        correDataSetPointGroups, alignedDataSetPointGroups, sRtParams = alignTestPointGroups(refPG, sortedToothPointGroups)#多进程执行CPD配准，寻找对应点对

        saveAlignedPointGroups2Txt(alignedDataSetPointGroups, sortedFileTags, saveDir)#保存对齐的下采样的点云，删除上次保存的

        fileHDF5 = os.path.join(dataSetDir, "cpdGpParams", "sRtParams_{}.hdf5".format(TOOTH_INDEX)) #保存s,R,t和对应顺序的tags
        saveRegistrationParams2Hdf5(fileHDF5, TOOTH_INDEX, sRtParams, tags=sortedFileTags)
        tags, scales, rotMats, tranVecs = readRegistrationParamsFromHDF5(fileHDF5, TOOTH_INDEX)

        printPlotRegistrationParams(sRtParams, plot=False)#画直方图


        # 计算误差
        eigVal, eigVec, A, meanTrainPointVector = getEigValVecOfSSMByPCA(alignedPointGroups) #主成分分析协方差矩阵
        numPC2Keep = 20 #需要使用的主成分数量
        featureVectors = A @ eigVec[:,:numPC2Keep]
        print("Num of PCA component used: {}, cumulative explained variance:{:.4f}%".format(numPC2Keep,100*remainedInfoRatio(eigVal, numPC2Keep))) # 保留前?个主成分

        alignedTestVectors = np.array([pg.flatten() for pg in alignedDataSetPointGroups]) # shape=(testSampleNum, 3*pointNum)
        normalizedTestPointVectors = alignedTestVectors - meanTrainPointVector
        featureVecs = normalizedTestPointVectors @ eigVec[:,:numPC2Keep]

        reconAlignedTestPointVectors = (featureVecs @ eigVec[:,:numPC2Keep].T + meanTrainPointVector) #重建的缩放刚性变换得到的测试点云
        reconAlignedTestPGs = [x.reshape(-1,3) for x in reconAlignedTestPointVectors] # 还需要inv平移，inv旋转，inv缩放
        reconInitTestPGs = [1./s * (alignedTestPG - t) @ np.linalg.inv(R) for s,R,t,alignedTestPG in zip(scales,rotMats, tranVecs, reconAlignedTestPGs)]#逆缩放刚性变换得到重建的原始的点云



        testSavePath = os.path.join(saveDir,"reconstruction")
        if not os.path.exists(testSavePath):
            os.makedirs(testSavePath)
        # 保存重建的测试点云（按配准参数进行复位）
        for testTag,pg,correPg in zip(sortedFileTags, reconInitTestPGs, correDataSetPointGroups):
            np.savetxt(os.path.join(testSavePath,"{}PC_recon_{}.txt".format(numPC2Keep, testTag)), pg)
            np.savetxt(os.path.join(testSavePath,"corre_init_{}.txt".format(testTag)), correPg)
        
        # 测试点云误差
        pointErrors = np.array([x-xPred for x,xPred in zip(correDataSetPointGroups, reconInitTestPGs)])
        pointDists = np.linalg.norm(pointErrors, axis=2, ord=2)

        print("Mean Corresponding Point Distance: {:.4f} mm".format(pointDists.mean()))
        print("Max Corresponding Point Distance: {:.4f} mm".format(pointDists.max()))
        print("Min Corresponding Point Distance: {:.4f} mm".format(pointDists.min()))

        initCenteredTestPGs = np.array([x-x.mean(axis=0) for x in correDataSetPointGroups])
        point2CentroidDists = np.linalg.norm(initCenteredTestPGs, axis=2, ord=2)
        relPointDists = pointDists / point2CentroidDists
        print("Mean Corresponding Point Relative Distance: {:.4f} ".format(relPointDists.mean()))
        print("Max Corresponding Point Relative Distance: {:.4f} ".format(relPointDists.max()))
        print("Min Corresponding Point Relative Distance: {:.4f} ".format(relPointDists.min()))

        log.close()



    






