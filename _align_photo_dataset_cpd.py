import cycpd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import shutil
import open3d as o3d
import time
import ray
import psutil
import h5py
from utils import fixedNumDownSample, getEigValVecOfSSMByPCA, saveAlignedPointGroups2Txt, loadAlignedPointGroupsWithIndex,\
    remainedInfoRatio, saveRegistrationParams2Hdf5, readRegistrationParamsFromHDF5, printPlotRegistrationParams





def getCorrePointPairs(probabilityMatrix):
    """根据cpd的概率矩阵，获取对应点对"""
    """matProb.shape=(pMovNum,pRefNum)"""
    matProb = probabilityMatrix.copy()
    pointPairs = []
    pRefNum = matProb.shape[1]
    
    # Method 1
    for i in range(pRefNum): #顺序贪婪（快）
        j = np.argmax(matProb[:,i])
        pointPairs.append((i, j)) # i:ref index, j: mov index
        matProb[j,:] = 0.0
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
def alignPointsByRigidRegistration(pRef, pMov, step, max_step, tolerance=1e-4, max_iter=100, num_cpus=4):
    """对两组点云缩放刚性配准，以点数较少的为参照，获取刚性对齐的具有相同数量的移动点云"""
    # X = zeroCentered(pRef)
    # Y = zeroCentered(pMov)
    X = pRef
    Y = pMov
    # reg =pycpd.RigidRegistration(**{'X': X, 'Y': Y, 'max_iterations':max_iter,'tolerance':tolerance})
    reg = cycpd.rigid_registration(**{'X': X, 'Y': Y, 'max_iterations':max_iter,'tolerance':tolerance,'verbose':False,'print_reg_params':False})
    TY,(s,r,t) = reg.register()
    # 第二次自由形变配准用于得到更加准确的对应点对
    reg2 = cycpd.deformable_registration(**{'X': X, 'Y': TY, 'max_iterations':max_iter//2,'tolerance':tolerance,'verbose':False,'print_reg_params':False})
    __deformedTY,(__G, __W) = reg2.register()
    pointPairs = getCorrePointPairs(reg2.P)
    correTY = extractCorreMovPoints(TY, pointPairs)
    correY = extractCorreMovPoints(Y, pointPairs)
    if step % num_cpus == 0:
        print("---------- Finish {}/{} ----------".format(step, max_step))
    return correY, correTY, TY, (s,r,t)

def normalizePointArray(X, scale, meanCentroid, rotMat=np.identity(3)):
    xCentroid = X.mean(axis=0)
    return (X-xCentroid)*scale @ rotMat + meanCentroid

def alignDataSetToothPointGroups(pRef, datasetPointGroups):
    """对dataWithPhoto数据集中的点云进行配准"""

    alignedPointGroups = []
    pGroupNum = len(datasetPointGroups)

    print("Start preliminairy alignment")
    remotes = []
    for i,pMov in enumerate(datasetPointGroups):
        rem = alignPointsByRigidRegistration.remote(pRef, pMov, step=i+1, max_step=pGroupNum, tolerance=1e-4, max_iter=100) 
        remotes.append(rem)
    remoteValues = ray.get(remotes) # [(correPMov, correTPMov, TPMov, param), ...]
    tempCorrePointGroups = [remoteVal[0] for remoteVal in remoteValues]

    # 最后一次CPD配准，将前一次下采样的得到的点云alignedPointGroups直接进行配准
    time.sleep(0.5)
    print("Start alignment of corresponding points")
    remotes = []
    for i,pMov in enumerate(tempCorrePointGroups):
        rem = alignPointsByRigidRegistration.remote(pRef, pMov, step=i+1, max_step=pGroupNum, tolerance=1e-4, max_iter=100) 
        remotes.append(rem)
    remoteValues = ray.get(remotes) # [(correPMov, correTPMov, TPMov, param), ...]
    correPointGroups = [remoteVal[0] for remoteVal in remoteValues]
    alignedPointGroups = [remoteVal[1] for remoteVal in remoteValues]
    params = [remoteVal[3] for remoteVal in remoteValues]

    return alignedPointGroups, correPointGroups, params



def getPointGroupByTag(pointGroupList, tagList, searchTag):
    for tag,pg in zip(tagList,pointGroupList):
        if tag == searchTag:
            return pg
    print("{} does not exist.".format(searchTag))
    return None



if __name__ == "__main__":
    
            
    num_cpus = psutil.cpu_count(logical=False) #ray(多线程)初始化
    ray.init(num_cpus=num_cpus, num_gpus=1)

    
    # TOOTH_INDICES = [11,12,13,14,15,16,17,21,22,23,24,25,26,27,31,32,33,34,35,36,37,41,42,43,44,45,46,47]
    # for toothIndex in TOOTH_INDICES:
    toothIndex = 11
    srcDir = r"./data/cpdAlignedData/{}/".format(toothIndex)
    alignedPointGroups, alignedPgTags = loadAlignedPointGroupsWithIndex(srcDir) #读取已经CPD配准的牙齿点云
    meanAlignedPG = np.array(alignedPointGroups).mean(axis=0)
    refPG = meanAlignedPG

    print("Align Tooth Index: {} in Data Set".format(toothIndex))

    dataSetDir = r"./dataWithPhoto"
    saveDir = os.path.join(dataSetDir, "cpdAlignedData", str(toothIndex))
    toothDir = os.path.join(dataSetDir, "ssa-repaired-txt", str(toothIndex))

    sortedToothPointGroups, sortedTxtFiles = getSortedToothPoints(toothDir) #参考点云按照点云中点数量从小到大排列
    sortedFileTags = [os.path.basename(f).split('.')[0] for f in sortedTxtFiles] # 排列的文件序号
    

    sortedToothPointGroups = [fixedNumDownSample(x,desiredNumOfPoint=3000, leftVoxelSize=1.0, rightVoxelSize=0.001) if x.shape[0]>3000 else x for x in sortedToothPointGroups] #对点数量较大的点云进行下采样加速,最多3000点
    print("max point num: ",sortedToothPointGroups[-1].shape)
    print("min point num: ",sortedToothPointGroups[0].shape)
    assert refPG.shape[0] <= sortedToothPointGroups[0].shape[0] #参考点云的点数量必须是最少的
    


    # TRAIN ALIGNMENT PROCESS (多线程)
    alignedDataSetPointGroups, correDataSetPointGroups, sRtParams = alignDataSetToothPointGroups(refPG, sortedToothPointGroups)#多进程执行CPD配准，寻找对应点对

    saveAlignedPointGroups2Txt(alignedDataSetPointGroups, sortedFileTags, saveDir)#保存对齐的下采样的点云，删除上次保存的

    fileHDF5 = os.path.join(dataSetDir, "params", "sRtParams_{}.hdf5".format(toothIndex)) #保存s,R,t和对应顺序的tags
    saveRegistrationParams2Hdf5(fileHDF5, toothIndex, sRtParams, tags=sortedFileTags)
    tags, scales, rotMats, tranVecs = readRegistrationParamsFromHDF5(fileHDF5, toothIndex)

    printPlotRegistrationParams(sRtParams, plot=False)#画直方图


    # 计算误差
    eigVal, eigVec, A, meanTrainPointVector = getEigValVecOfSSMByPCA(alignedPointGroups) #主成分分析协方差矩阵
    numPC2Keep = 60 #需要使用的主成分数量
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



    






