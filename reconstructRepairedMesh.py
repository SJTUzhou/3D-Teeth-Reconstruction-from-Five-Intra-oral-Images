import cycpd
import matplotlib.pyplot as plt
import pycpd
import numpy as np
import pandas as pd
import os
import glob
import shutil
import open3d as o3d
import time
import h5py
import copy
import trimesh
import ray
import psutil
from pcd_mesh_utils import fixedNumDownSample, surfaceVertices2WatertightO3dMesh, loadAlignedPointGroupsWithIndex, getEigValVecOfSSMByPCA, remainedInfoRatio


IndicesOfScanPgRepairedBySSM = {11:[15,18,22,25,37,40,55,58,59,68,73,83,89,91],
                      12:[31,36,61,68,69],
                      13:[0,22,35,45,58,65,68,77,89], 
                      14:[12,31,35,54,63,69,77],
                      15:[2,12,26,52,65,74], 
                      16:[2,14,47,50,51,54,63,86],
                      17:[32,33,64,74,81,86,93],
                      21:[6,15,18,20,24,37,40,62,85,89,91],
                      22:[6,15,25,34,48,49,56,59,80],
                      23:[7,20,38,41,49,57,75],
                      24:[12,23,30,38,48,60,75,82,84,89],
                      25:[3,23,29,38,42,60,76,79,84],
                      26:[3,9,16,19,27,28,29,42,46,72,87,88,90,92],
                      27:[1,9,16,21,27,29,39,46,66,72,87,88,94],
                      31:[23,25,28,48,53,55,58,75,90,93],
                      32:[4,25,34,52,90,93],
                      33:[1,25,26,52],
                      34:[34,54,69],
                      35:[26,29,44,54,56,69,71,76],
                      36:[2,27,30,33,38,47,49,56,58,71,73,86,94],
                      37:[14,16,17,27,30,38,47,49,50,51,72,84,90,94],
                      41:[10,19,23,59,62,75,80,81],
                      42:[3,12,19,32,58,64],
                      43:[7,8,12,19,24,41,57,59,64],
                      44:[40,41,61,76,80],
                      45:[0,13,37,40,46,61,66,76,79],
                      46:[0,5,15,35,36,37,40,42,45,48,66,67,68,70,77,79,85],
                      47:[13,21,22,35,36,45,48,60,63,68,70,83,92]}

TagsOfScanPgRepairedBySSM = {k:["{}{}".format(id,"U") if k < 30 else "{}{}".format(id,"L") for id in v] for k,v in IndicesOfScanPgRepairedBySSM.items()}


def getUpSampledScanPG(scanMesh, desiredNumOfPoint=2500):
    v, f = trimesh.remesh.subdivide_to_size(vertices=np.asarray(scanMesh.vertices), faces=np.asarray(scanMesh.triangles), max_edge=0.2, max_iter=10, return_index=False)
    v = fixedNumDownSample(v, desiredNumOfPoint=desiredNumOfPoint, leftVoxelSize=1.0, rightVoxelSize=0.01)
    return v.astype(np.double)

def computeTX(X, s, R, t): #计算反向移动的测试点云
    return 1./s * (X - t) @ R.T

def computeX(TX, s, R, t): #计算反向移动的测试点云
    return s * TX @ R + t

def getCorrePointPairsWithMinProb(probabilityMatrix, minProb):
    """根据cpd的概率矩阵，获取对应点对"""
    """matProb.shape=(pMovNum,pRefNum)"""
    matProb = probabilityMatrix.copy()
    pointPairs = []
    pRefNum = matProb.shape[1]
    for i in range(pRefNum): #顺序贪婪（快）
        j = np.argmax(matProb[:,i])
        if matProb[j,i] > minProb:
            pointPairs.append((i, j)) # i:ref index, j: mov index
            matProb[j,:] = 0.0
    return np.array(pointPairs, dtype=np.uint32)

def extractCorreMovPoints(pMov, pointPairs):
    return pMov[pointPairs[:,1], :]

def extractCorreRefPoints(pRef, pointPairs):
    return pRef[pointPairs[:,0], :]

def findOptimalMinProb(matProb, priorPNum, priorMeanMeshArea, testBrokenMeshArea, scaleFactor):
    """找到最优的概率阈值使得 priorMeanMesh的选中区域的面积 * scaleFactor^2 == testBrokenMesh的面积"""
    leftProb = 1.0
    rightProb = 0.0
    midProb = (leftProb + rightProb) / 2.
    priorArea = priorMeanMeshArea
    obserArea = testBrokenMeshArea
    pNum = getCorrePointPairsWithMinProb(matProb, midProb).shape[0]
    expectedPNum = obserArea / (priorArea * scaleFactor**2) * priorPNum
    assert expectedPNum < matProb.shape[0] and expectedPNum < matProb.shape[1]
    # print("expected num of point in test broken pointcloud: ",expectedPNum)
    while np.abs(pNum - expectedPNum) > 1:
        if pNum < expectedPNum:
            leftProb = copy.copy(midProb)
        else:
            rightProb = copy.copy(midProb)
        midProb = (leftProb + rightProb) / 2.
        pNum = getCorrePointPairsWithMinProb(matProb, midProb).shape[0]
    return midProb

def alignScanPG2PriorMeanPG(scanPG, scanMeshArea, priorMeanPG, priorMeanMeshArea, max_iter=100, tol=1e-4):
    """将扫描牙齿点云（存在缺损）与先验平均点云配准，找到对应点"""
    X = scanPG #为提高配准精度，这里将扫描牙齿点云（存在缺损）作为参考点云，先验平均点云作为移动点云
    Y = priorMeanPG
    reg = cycpd.rigid_registration(**{'X': X, 'Y': Y,'max_iterations':max_iter,'tolerance':tol,'verbose':False,'print_reg_params':False}) #缩放刚性配准确定缩放尺寸
    TY,(s,r,t) = reg.register()
    reg2 = cycpd.affine_registration(**{'X': X, 'Y': TY,'max_iterations':max_iter//2,'tolerance':tol,'verbose':False,'print_reg_params':False}) #仿射变换配准，保证牙齿轮廓边缘相匹配
    TTY,(_B,_t) = reg2.register()
    optimalInfProb = findOptimalMinProb(reg2.P, priorMeanPG.shape[0], priorMeanMeshArea, scanMeshArea, s)
    pointPairs = getCorrePointPairsWithMinProb(reg2.P, minProb=optimalInfProb)
    pointPairsOrder = sorted(range(len(pointPairs)), key=lambda x:pointPairs[x,1]) # 按照先验平均点云的index进行排序
    pointPairs = pointPairs[pointPairsOrder]
    # 假设以Y为参考点云，对应变换X
    TX = computeTX(X, s, r, t)
    correTX = extractCorreRefPoints(TX, pointPairs)
    correY = extractCorreMovPoints(Y, pointPairs)
    return pointPairs, correTX, correY, (s,r,t)

def getObservedIndices(pointPairs):
    obserIndices = np.vstack([3*pointPairs[:,1], 3*pointPairs[:,1]+1, 3*pointPairs[:,1]+2]).T.flatten()
    return obserIndices

def predictFeatureVecFromSSM(correAlignedScanPG, correPriorMeanPG, priorMeanPG, eigVec, numPC2Keep, obserIndices, lambd=0.05):
    # subV @ featureVec = observedVec - observedPriorVecMu 最小二乘解featureVec
    V = eigVec[:,:numPC2Keep]
    subV = eigVec[obserIndices,:numPC2Keep]
    # lambd: a small constant to avoid over-fitting while maintaining acceptable residuals
    featureVec = np.linalg.inv(subV.T @ subV + lambd*np.identity(numPC2Keep)) @ subV.T @ (correAlignedScanPG - correPriorMeanPG).flatten()
    predAlignedRepairedPG = priorMeanPG + (featureVec @ V.T).reshape(-1,3) 
    return featureVec, predAlignedRepairedPG

def predictRepairedPG(predAlignedRepairedPG, s, r, t):
    predRepairedPG = s * predAlignedRepairedPG @ r + t
    return predRepairedPG

def removeOutliers(vertices, nb_points=10, radius=0.8):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd = pcd.select_by_index(ind, invert=False)
    return np.asarray(pcd.points)


@ray.remote
def parallelRepairMesh(scanPG, scanMeshArea, priorMeanPG, priorMeanMeshArea, numPC2Keep, dstTxtDir, dstObjDir, fileTag, lambd):
    pointPairs, correAlignedScanPG, correPriorMeanPG, (s,r,t) = alignScanPG2PriorMeanPG(scanPG, scanMeshArea, priorMeanPG, priorMeanMeshArea, max_iter=500, tol=1e-4)
    obserIndices = getObservedIndices(pointPairs)
    featureVec, predAlignedRepairedPG = predictFeatureVecFromSSM(correAlignedScanPG, correPriorMeanPG, priorMeanPG, eigVec, numPC2Keep, obserIndices, lambd=lambd)
    predRepairedPG = predictRepairedPG(predAlignedRepairedPG, s, r, t)

    repairedPG = removeOutliers(predRepairedPG, nb_points=2, radius=0.5)
    repairedMesh = surfaceVertices2WatertightO3dMesh(repairedPG) # 生成水密网格
    saveObjName = os.path.join(dstObjDir, fileTag+".obj")
    o3d.io.write_triangle_mesh(saveObjName, repairedMesh, write_ascii=False, compressed=False, write_vertex_normals=False,\
         write_vertex_colors=False, write_triangle_uvs=False, print_progress=False)
    repairedPG = np.asarray(repairedMesh.vertices)

    saveTxtName = os.path.join(dstTxtDir, fileTag+".txt")
    np.savetxt(saveTxtName, repairedPG)
    print("Finish repairing ", saveTxtName)
    

def getFileTag(f):
    return os.path.basename(f).split(".")[0]


if __name__ == "__main__":
    num_cpus = psutil.cpu_count(logical=False) #ray(多线程)初始化
    ray.init(num_cpus=num_cpus, num_gpus=1)

    lambd = 0.0
    toothIndex = 37

    loadDir = r"./data/cpdAlignedData/{}/".format(toothIndex)
    alignedPointGroups, alignedPgTags = loadAlignedPointGroupsWithIndex(loadDir)
    eigVal, eigVec, A, meanTrainPointVector = getEigValVecOfSSMByPCA(alignedPointGroups)
    priorMeanPG = meanTrainPointVector.reshape(-1,3).astype(np.double)
    priorMeanMesh = surfaceVertices2WatertightO3dMesh(priorMeanPG)
    priorMeanMeshArea = priorMeanMesh.get_surface_area()

    numPC2Keep = 60 # 90% cumulative explained variance
    print("Use {:.2f}% cumulative explained variance for reconstruction".format(100*remainedInfoRatio(eigVal,numPC2Keep)))
    scanMeshDir = r"./dataWithPhoto/format-obj/{}/".format(toothIndex)
    dstTxtDir = r"./dataWithPhoto/repaired-txt/{}/".format(toothIndex)
    dstObjDir = r"./dataWithPhoto/repaired-obj/{}/".format(toothIndex)
    if not os.path.exists(dstTxtDir):
        os.makedirs(dstTxtDir)
    if not os.path.exists(dstObjDir):
        os.makedirs(dstObjDir)
    
    # allScanMeshObjs = glob.glob(os.path.join(scanMeshDir,"*.obj"))
    # allScanMeshObjs = sorted(allScanMeshObjs, key=lambda x:int(getFileTag(x)[:-1]))

    # allFileTags = [getFileTag(f) for f in allScanMeshObjs]

    fileTags = TagsOfScanPgRepairedBySSM[toothIndex] # 需要用SSM修复的tag
    scanMeshObjs = [os.path.join(scanMeshDir,tag+".obj") for tag in fileTags]
    print("{}: {}".format(toothIndex, fileTags))

    scanMeshAreas = []
    scanPGs = []
    for objF in scanMeshObjs:
        scanMesh = o3d.io.read_triangle_mesh(objF)
        scanMeshAreas.append(scanMesh.get_surface_area())
        scanPGs.append(getUpSampledScanPG(scanMesh, desiredNumOfPoint=2500))
        # print("Finish up-sampling ", objF)
    
    print("Start reconstruction of repaired mesh for tooth index: ", toothIndex)
    ray.get([parallelRepairMesh.remote(scanPG, scanMeshArea, priorMeanPG, priorMeanMeshArea, numPC2Keep, dstTxtDir, dstObjDir, fileTag, lambd)\
        for scanPG, scanMeshArea, fileTag in zip(scanPGs, scanMeshAreas, fileTags)])

        