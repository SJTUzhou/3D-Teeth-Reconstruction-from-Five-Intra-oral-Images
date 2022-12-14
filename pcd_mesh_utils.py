import open3d as o3d
import trimesh
import numpy as np
import copy




def showPointCloud(vertices, windowName):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.visualization.draw_geometries([pcd], window_name=windowName, width=800, height=600, left=50,top=50, point_show_normal=False)
    

def fixedNumDownSample(vertices, desiredNumOfPoint, leftVoxelSize, rightVoxelSize):
    # use bi-section to find a voxel_size that we can get the desiredNumOfPoint after down sampling
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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(voxel_size)
    downSampledVertices = np.asarray(pcd.points, dtype=np.double)
    return downSampledVertices



def farthestPointDownSample(vertices, num_point_sampled, return_flag=False):
    # vertices.shape = (N,3) or (N,2)
    N = len(vertices)
    n = num_point_sampled
    assert n <= N, "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.mean(vertices, axis=0) # centroid of vertices
    _d = np.linalg.norm(vertices - _G, axis=1, ord=2)
    farthest = np.argmax(_d) 
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_) 
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
    """ Get the largest connected vertices and faces in a triangle mesh
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





def surfaceVertices2WatertightO3dMesh(vertices, showInWindow=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals()
    # to obtain a consistent normal orientation
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # surface reconstruction using Poisson reconstruction
    __mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, scale=1.1)
    if showInWindow == True:
        __mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))
        o3d.visualization.draw_geometries([__mesh], window_name='Open3D reconstructed watertight mesh', width=800, height=600, left=50,
                                          top=50, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True)
    return __mesh




def mergeO3dTriangleMeshes(o3dMeshes):
    if len(o3dMeshes) == 0:
        return []
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



