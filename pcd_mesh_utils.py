# Utility functions to process pointcloud and triangle mesh

import copy

import cycpd
import numpy as np
import open3d as o3d
import trimesh


def showPointCloud(vertices, windowName=""):
    """Visualize the pointcloud given the 3D points
    vertices: numpy array, shape (N,3)"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=windowName,
        width=800,
        height=600,
        left=50,
        top=50,
        point_show_normal=False,
    )


def farthestPointDownSample(vertices, num_point_sampled, return_flag=False):
    """Farthest Point Sampling (FPS) algorithm
    Input:
        vertices: numpy array, shape (N,3) or (N,2)
        num_point_sampled: int, the number of points after downsampling, should be no greater than N
        return_flag: bool, whether to return the mask of the selected points
    Output:
        selected_vertices: numpy array, shape (num_point_sampled,3) or (num_point_sampled,2)
        [Optional] flags: boolean numpy array, shape (N,3) or (N,2)"""
    N = len(vertices)
    n = num_point_sampled
    assert (
        n <= N
    ), "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.mean(vertices, axis=0)  # centroid of vertices
    _d = np.linalg.norm(vertices - _G, axis=1, ord=2)
    farthest = np.argmax(_d)
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_)
    for i in range(n):
        flags[farthest] = True
        distances[farthest] = 0.0
        p_farthest = vertices[farthest]
        dists = np.linalg.norm(vertices[~flags] - p_farthest, axis=1, ord=2)
        distances[~flags] = np.minimum(distances[~flags], dists)
        farthest = np.argmax(distances)
    if return_flag == True:
        return vertices[flags], flags
    else:
        return vertices[flags]


def surfaceVertices2WatertightO3dMesh(vertices, showInWindow=False):
    """Construct single tooth triangle mesh from surface vertices by Poisson surface reconstruction
    Input:
        vertices: numpy array, shape (N,3)
        showInWindow: bool, whether to visualize the constructed mesh
    Output:
        __mesh: open3d.geometry.TriangleMesh, the constructed mesh of a single tooth
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals()
    # to obtain a consistent normal orientation
    pcd.orient_normals_consistent_tangent_plane(k=30)
    pcd.normalize_normals()
    normal = np.asarray(pcd.normals)
    center = np.mean(vertices, 0)
    vecs = vertices - center
    opposite_mask = np.sum(vecs * normal, 1) < 0
    normal[opposite_mask] = -normal[opposite_mask]
    pcd.normals = o3d.utility.Vector3dVector(normal)
    # surface reconstruction using Poisson reconstruction
    __mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, scale=1.1
    )
    if showInWindow == True:
        __mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))
        o3d.visualization.draw_geometries(
            [__mesh],
            window_name="Open3D reconstructed watertight mesh",
            width=800,
            height=600,
            left=50,
            top=50,
            point_show_normal=True,
            mesh_show_wireframe=True,
            mesh_show_back_face=True,
        )
    return __mesh


def exportTriMeshObj(vertices, faces, objFile):
    """Save a naive triangle mesh in OBJ format
    Input:
        vertices: numpy array, shape (N,3)
        faces: numpy array, shape (N,3)
        objFile: str, file path to save with ".obj" suffix
    """
    __mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    exportStr = trimesh.exchange.obj.export_obj(
        __mesh,
        include_normals=False,
        include_color=False,
        include_texture=False,
        return_texture=False,
        write_texture=False,
        resolver=None,
        digits=8,
    )
    with open(objFile, "w") as f:
        f.write(exportStr)


def mergeO3dTriangleMeshes(o3dMeshes):
    """Merge open3d triangle meshes
    Input:
        o3dMeshes: List of open3d.geometry.TriangleMesh
    Output:
        aggMesh: open3d.geometry.TriangleMesh
    """
    assert len(o3dMeshes) > 0
    aggMesh = copy.deepcopy(o3dMeshes[0])
    for _mesh in o3dMeshes[1:]:
        aggMesh += _mesh
    return aggMesh


def getAlignedSrcPointCloud(X_src, X_target, with_scale=False):
    """registration between two pointclouds, first downsampling then use CPD
    Input:
        X_src: numpy array, shape (M,3)
        X_target: numpy array, shape (?,3)
        with_scale: bool, True for similarity registration, False for rigid regis.
    Output:
        aligned X_src: numpy array, shape (M,3)"""
    assert X_src.ndim == 2, "X_src array should be 2d."
    assert X_target.ndim == 2, "X_target array should be 2d."
    ds_X_src = farthestPointDownSample(X_src, num_point_sampled=3000, return_flag=False)
    ds_X_target = farthestPointDownSample(
        X_target, num_point_sampled=3000, return_flag=False
    )
    ds_X_src = ds_X_src.astype(np.double)
    ds_X_target = ds_X_target.astype(np.double)
    reg = cycpd.rigid_registration(
        **{
            "X": ds_X_target,
            "Y": ds_X_src,
            "scale": with_scale,
            "max_iterations": 100,
            "tolerance": 0.1,
            "w": 0,
            "verbose": True,
            "print_reg_params": True,
        }
    )
    _, (s, r, t) = reg.register()
    return s * np.dot(X_src, r) + t
