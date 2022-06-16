import open3d as o3d
import trimesh
from trimesh.voxel import creation as tri_creation
import numpy as np
import scipy
import copy
import glob
import shutil
import os
import h5py
import matplotlib.pyplot as plt
import ray

UPPER_INDICES = [11,12,13,14,15,16,17,21,22,23,24,25,26,27] #不考虑智齿18,28
LOWER_INDICES = [31,32,33,34,35,36,37,41,42,43,44,45,46,47] #不考虑智齿38,48


def fixedNumDownSample(vertices, desiredNumOfPoint, leftVoxelSize, rightVoxelSize):
    # 二分法寻找最佳voxel_size使得下采样后的点云中的点数量恰好为desiredNumOfPoint
    assert leftVoxelSize > rightVoxelSize, "leftVoxelSize should be larger than rightVoxelSize"
    assert vertices.shape[0] >= desiredNumOfPoint, "desiredNumOfPoint should be less than or equal to the num of points in the given array."
    if vertices.shape[0] == desiredNumOfPoint:
        return vertices
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(leftVoxelSize)
    assert len(pcd.points) <= desiredNumOfPoint, "Please specify a larger leftVoxelSize."
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(rightVoxelSize)
    assert len(pcd.points) >= desiredNumOfPoint, "Please specify a smaller rightVoxelSize."
    
    pcd.points = o3d.utility.Vector3dVector(vertices)
    midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
    pcd = pcd.voxel_down_sample(midVoxelSize)
    iterCount = 0
    while len(pcd.points) != desiredNumOfPoint:
        if len(pcd.points) < desiredNumOfPoint:
            leftVoxelSize = copy.copy(midVoxelSize)
        else:
            rightVoxelSize = copy.copy(midVoxelSize)
        midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd = pcd.voxel_down_sample(midVoxelSize)
        iterCount += 1
        if iterCount > 128:
            diffNum = len(pcd.points) - desiredNumOfPoint
            if diffNum > 0:
                selectedMask = np.ones((len(pcd.points),), dtype=np.bool_)
                selectedMask[np.random.randint(0,len(pcd.points),size=diffNum)] = False
                downSampledVertices = np.asarray(pcd.points, dtype=vertices.dtype)[selectedMask,:]
                assert downSampledVertices.shape[0] == desiredNumOfPoint
                return downSampledVertices
            else:
                rightVoxelSize = rightVoxelSize - 1e-4
    # print("final voxel size: ", midVoxelSize)
    downSampledVertices = np.asarray(pcd.points, dtype=vertices.dtype)
    assert downSampledVertices.shape[0] == desiredNumOfPoint
    return downSampledVertices

def voxelDownSample(vertices, voxel_size):
    # 体素均匀下采样
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(voxel_size)
    downSampledVertices = np.asarray(pcd.points, dtype=np.double)
    return downSampledVertices



def farthestPointDownSample(vertices, num_point_sampled, return_flag=False):
    # 最远点采样 FPS # vertices.shape = (N,3) or (N,2)
    N = len(vertices)
    n = num_point_sampled
    assert n <= N, "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.mean(vertices, axis=0) # centroid of vertices
    _d = np.linalg.norm(vertices - _G, axis=1, ord=2)
    farthest = np.argmax(_d) # 取离重心最远的点为起始点
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_) # 点是否被选中
    for i in range(n):
        flags[farthest] = True
        distances[farthest] = 0.
        p_farthest = vertices[farthest]
        dists = np.linalg.norm(vertices[~flags] - p_farthest, axis=1, ord=2)
        distances[~flags] = np.minimum(distances[~flags], dists)
        farthest = np.argmax(distances)
    if return_flag == True:
        return vertices[flags], flags
    else:
        return vertices[flags]


def getLargestConnectedMeshComponent(vertices, faces):
    """ 获取三角面片中的最大连通区域
        Get the largest connected vertices and faces in a triangle mesh
        INPUT:
            vertices: numpy array, shape(n,3), float64
            faces: numpy array, shape(n,3), int32
        RETURN: newVertices, newFaces
    """
    __mesh = o3d.geometry.TriangleMesh()
    __mesh.vertices = o3d.utility.Vector3dVector(vertices)
    __mesh.triangles = o3d.utility.Vector3iVector(faces)
    triangle_clusters, cluster_n_triangles, cluster_area = (__mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    # print("Number of triangle clusters: ", len(cluster_n_triangles))
    # print("triangle number in clusters: ", cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    goodConnectionFlag = np.sum(cluster_n_triangles) < 1.2*cluster_n_triangles[largest_cluster_idx]
    __mesh.remove_triangles_by_mask(triangles_to_remove)
    __mesh.remove_unreferenced_vertices()
    return np.asarray(__mesh.vertices, dtype=np.float32), np.asarray(__mesh.triangles, dtype=np.int32), goodConnectionFlag


def removeMeshSmallComponents(vertices, faces, minNumVertex2Keep=100):
    __mesh = o3d.geometry.TriangleMesh()
    __mesh.vertices = o3d.utility.Vector3dVector(vertices)
    __mesh.triangles = o3d.utility.Vector3iVector(faces)
    triangle_clusters, cluster_n_triangles, cluster_area = (__mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    # print("Number of triangle clusters: ", len(cluster_n_triangles))
    # print("triangle number in clusters: ", cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < minNumVertex2Keep
    __mesh.remove_triangles_by_mask(triangles_to_remove)
    __mesh.remove_unreferenced_vertices()
    return np.asarray(__mesh.vertices, dtype=np.float32), np.asarray(__mesh.triangles, dtype=np.int32)


def showPointCloud(vertices, windowName):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.visualization.draw_geometries([pcd], window_name=windowName, width=800, height=600, left=50,top=50, point_show_normal=False)

    
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def surfaceVertices2WatertightO3dMesh(vertices, showInWindow=False):
    """根据点云表面点重建水密的三角面片,三角面片顶点数量多于原始点数量"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals()
    # to obtain a consistent normal orientation
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # surface reconstruction using Poisson reconstruction
    __mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, scale=1.1)
    if showInWindow == True:
        __mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))
        o3d.visualization.draw_geometries([__mesh], window_name='Open3D reconstructed watertight mesh', width=800, height=600, left=50,
                                          top=50, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True)
    return __mesh

def mergeO3dTriangleMeshes(o3dMeshes):
    """合并多个Triangle Meshes"""
    newVStartIndex = 0
    aggVertices = []
    aggTriangles = []
    for _mesh in o3dMeshes:
        vNum = len(_mesh.vertices)
        tempVertices = np.asarray(_mesh.vertices)
        tempTriangles = np.asarray(_mesh.triangles) + newVStartIndex
        aggVertices.append(tempVertices)
        aggTriangles.append(tempTriangles)
        newVStartIndex += vNum
    aggMesh = o3d.geometry.TriangleMesh()
    aggMesh.vertices = o3d.utility.Vector3dVector(np.vstack(aggVertices))
    aggMesh.triangles = o3d.utility.Vector3iVector(np.vstack(aggTriangles))
    return aggMesh

def exportTriMeshObj(vertices, faces, objFile):
    __mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    exportStr = trimesh.exchange.obj.export_obj(__mesh, include_normals=False, include_color=False, include_texture=False, 
        return_texture=False, write_texture=False, resolver=None, digits=8)
    with open(objFile,"w") as f:
        f.write(exportStr)
        
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity, ord="fro")
    detR = np.linalg.det(R)
    return (n < 1e-6) and (np.abs(detR-1) < 1e-6)

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(RotMat):
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
    """根据tags保存对应的sRtParams"""
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



def computeTransMatByCorres(X_src, X_target, with_scale=False):
    assert X_src.ndim == 2, "X_src array should be 2d."
    assert X_target.ndim == 2, "X_target array should be 2d."
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(X_src)
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(X_target)
    pNum = len(X_src)
    _corre = np.tile(np.arange(pNum),(2,1)).T
    corres = o3d.utility.Vector2iVector(_corre)
    reg = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scale)
    homoTransMat = reg.compute_transformation(pcd_src, pcd_target, corres)
    return homoTransMat.T


def computeRMSE(X_pred, X_Ref):
    # 计算 Root Mean Square Error of corresponding points
    pointL2Errors = np.linalg.norm(X_pred - X_Ref, axis=2, ord=2)
    return np.mean(pointL2Errors)

def computeRMSD(X_Pred, X_Ref):
    # 计算 Root Mean Squared symmetric surface Distance
    RMSDs = []
    for x_pred, x_ref in zip(X_Pred, X_Ref):
        dist_mat = scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8))
        squared_sd1 = np.square(np.min(dist_mat, axis=0))
        squared_sd2 = np.square(np.min(dist_mat, axis=1))
        rmsd = np.sqrt(np.mean(np.hstack([squared_sd1, squared_sd2])))
        RMSDs.append(rmsd)
    return np.mean(RMSDs)

def computeASSD(X_Pred, X_Ref):
    # 计算 Average Symmetric Surface Distance
    ASSDs = []
    for x_pred, x_ref in zip(X_Pred, X_Ref):
        dist_mat = scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8))
        sd1 = np.min(dist_mat, axis=0)
        sd2 = np.min(dist_mat, axis=1)
        assd = np.mean(np.hstack([sd1, sd2]))
        ASSDs.append(assd)
    return np.mean(ASSDs)

def computeHD(X_Pred, X_Ref):
    # 计算 Hausdorff Distance(豪斯多夫距离)
    HDs = []
    for x_pred, x_ref in zip(X_Pred, X_Ref):
        dist_mat = scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8))
        hd1 = np.max(np.min(dist_mat, axis=0))
        hd2 = np.max(np.min(dist_mat, axis=1))
        HDs.append(np.max([hd1,hd2]))
    return np.mean(HDs)


def computeDiceAndVOE(x_ref, x_pred, pitch=0.2):
    ''' compute volume dice coefficient and volumetric overlap error of two surface point clouds
        Assume the 2 surface point clouds are already aligned'''
    # convert surface point cloud to watertight mesh
    msh_ref_o3d = surfaceVertices2WatertightO3dMesh(x_ref, showInWindow=False)
    msh_pred_o3d = surfaceVertices2WatertightO3dMesh(x_pred, showInWindow=False)
    msh_ref_tri = trimesh.Trimesh(vertices=np.asarray(msh_ref_o3d.vertices), faces=np.asarray(msh_ref_o3d.triangles))
    msh_pred_tri = trimesh.Trimesh(vertices=np.asarray(msh_pred_o3d.vertices), faces=np.asarray(msh_pred_o3d.triangles))
    # voxelize two meshes in the same coordinate
    lbs = np.minimum(msh_ref_tri.bounds[0,:], msh_pred_tri.bounds[0,:])
    ubs = np.maximum(msh_ref_tri.bounds[1,:], msh_pred_tri.bounds[1,:])
    voxel_center = (lbs + ubs) / 2.
    margin = 2
    radius = int(np.max((ubs-lbs)/(2.*pitch))) + margin
    vxl_grid_ref = tri_creation.local_voxelize(msh_ref_tri, point=voxel_center, pitch=pitch, radius=radius, fill=True)
    vxl_grid_pred = tri_creation.local_voxelize(msh_pred_tri, point=voxel_center, pitch=pitch, radius=radius, fill=True)
    # Get 3d boolean array representation of the 2 voxel grids
    bool3d_ref = np.asarray(vxl_grid_ref.matrix, np.bool_)
    bool3d_pred = np.asarray(vxl_grid_pred.matrix, np.bool_)
    Intersection = np.logical_and(bool3d_ref, bool3d_pred).sum()
    Union = np.logical_or(bool3d_ref, bool3d_pred).sum()
    Dice = 2.*Intersection / (Intersection + Union)
    VOE = 1. - Intersection / Union
    return Dice, VOE
