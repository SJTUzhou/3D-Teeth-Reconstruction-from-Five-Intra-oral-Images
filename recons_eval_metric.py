# Evaluation Metrics for 3D teeth Reconstruction

import numpy as np
import scipy
import trimesh
from trimesh.voxel import creation as tri_creation

from pcd_mesh_utils import surfaceVertices2WatertightO3dMesh


def computeRMSD(X_Pred, X_Ref, return_list=False):
    """compute Root Mean Squared symmetric surface Distance between two tooth rows
    Input:
        X_Pred: List of numpy array, each array's shape = (?,3)
        X_Ref: List of numpy array, each array's shape = (?,3), len(X_Ref) == len(X_Pred)
        return_list: bool, whether to return the mean metric value for each tooth in the tooth row
    Output:
        mean RMSD, double if return_list == True else List of double
    """
    RMSDs = []
    assert len(X_Pred) == len(
        X_Ref
    ), "the number of teeth in each tooth row should be equal."
    for x_pred, x_ref in zip(X_Pred, X_Ref):
        dist_mat = scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8))
        squared_sd1 = np.square(np.min(dist_mat, axis=0))
        squared_sd2 = np.square(np.min(dist_mat, axis=1))
        rmsd = np.sqrt(np.mean(np.hstack([squared_sd1, squared_sd2])))
        RMSDs.append(rmsd)
    if return_list == True:
        return RMSDs
    return np.mean(RMSDs)


def computeASSD(X_Pred, X_Ref, return_list=False):
    """compute Average Symmetric Surface Distance between two tooth rows
    Input:
        X_Pred: List of numpy array, each array's shape = (?,3)
        X_Ref: List of numpy array, each array's shape = (?,3), len(X_Ref) == len(X_Pred)
        return_list: bool, whether to return the mean metric value for each tooth in the tooth row
    Output:
        mean ASSD, double if return_list == True else List of double
    """
    ASSDs = []
    assert len(X_Pred) == len(
        X_Ref
    ), "the number of teeth in each tooth row should be equal."
    for x_pred, x_ref in zip(X_Pred, X_Ref):
        dist_mat = scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8))
        sd1 = np.min(dist_mat, axis=0)
        sd2 = np.min(dist_mat, axis=1)
        assd = np.mean(np.hstack([sd1, sd2]))
        ASSDs.append(assd)
    if return_list == True:
        return ASSDs
    return np.mean(ASSDs)


def computeHD(X_Pred, X_Ref, return_list=False):
    """compute Hausdorff Distance between two tooth rows
    Input:
        X_Pred: List of numpy array, each array's shape = (?,3)
        X_Ref: List of numpy array, each array's shape = (?,3), len(X_Ref) == len(X_Pred)
        return_list: bool, whether to return the mean metric value for each tooth in the tooth row
    Output:
        mean HD, double if return_list == True else List of double
    """
    HDs = []
    assert len(X_Pred) == len(
        X_Ref
    ), "the number of teeth in each tooth row should be equal."
    for x_pred, x_ref in zip(X_Pred, X_Ref):
        dist_mat = scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8))
        hd1 = np.max(np.min(dist_mat, axis=0))
        hd2 = np.max(np.min(dist_mat, axis=1))
        HDs.append(np.max([hd1, hd2]))
    if return_list == True:
        return HDs
    return np.mean(HDs)


def computeChamferDistance(X_Pred, X_Ref, return_list=False):
    """compute Chamfer Distance between two tooth rows
    Input:
        X_Pred: List of numpy array, each array's shape = (?,3)
        X_Ref: List of numpy array, each array's shape = (?,3), len(X_Ref) == len(X_Pred)
        return_list: bool, whether to return the mean metric value for each tooth in the tooth row
    Output:
        mean CD, double if return_list == True else List of double
    """
    CDs = []
    assert len(X_Pred) == len(
        X_Ref
    ), "the number of teeth in each tooth row should be equal."
    for x_pred, x_ref in zip(X_Pred, X_Ref):
        squaredDistMat = (
            scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8)) ** 2
        )
        CDs.append(
            np.min(squaredDistMat, axis=0).mean()
            + np.min(squaredDistMat, axis=1).mean()
        )
    if return_list == True:
        return CDs
    return np.mean(CDs)


def computeDiceAndVOE(x_ref, x_pred, pitch=0.2):
    """compute volume dice coefficient and volumetric overlap error of two surface point clouds
        Assume the 2 surface point clouds are already aligned
    Input:
        x_pred: numpy array, shape = (?,3)
        x_ref: numpy array, shape = (?,3)
        pitch: double, controls the voxel size when voxelizing the mesh
    Output:
        double, double"""
    # convert surface point cloud to watertight mesh
    msh_ref_o3d = surfaceVertices2WatertightO3dMesh(x_ref, showInWindow=False)
    msh_pred_o3d = surfaceVertices2WatertightO3dMesh(x_pred, showInWindow=False)
    msh_ref_tri = trimesh.Trimesh(
        vertices=np.asarray(msh_ref_o3d.vertices),
        faces=np.asarray(msh_ref_o3d.triangles),
    )
    msh_pred_tri = trimesh.Trimesh(
        vertices=np.asarray(msh_pred_o3d.vertices),
        faces=np.asarray(msh_pred_o3d.triangles),
    )
    # voxelize two meshes in the same coordinate
    lbs = np.minimum(msh_ref_tri.bounds[0, :], msh_pred_tri.bounds[0, :])
    ubs = np.maximum(msh_ref_tri.bounds[1, :], msh_pred_tri.bounds[1, :])
    voxel_center = (lbs + ubs) / 2.0
    margin = 2
    radius = int(np.max((ubs - lbs) / (2.0 * pitch))) + margin
    vxl_grid_ref = tri_creation.local_voxelize(
        msh_ref_tri, point=voxel_center, pitch=pitch, radius=radius, fill=True
    )
    vxl_grid_pred = tri_creation.local_voxelize(
        msh_pred_tri, point=voxel_center, pitch=pitch, radius=radius, fill=True
    )
    # Get 3d boolean array representation of the 2 voxel grids
    bool3d_ref = np.asarray(vxl_grid_ref.matrix, np.bool_)
    bool3d_pred = np.asarray(vxl_grid_pred.matrix, np.bool_)
    Intersection = np.logical_and(bool3d_ref, bool3d_pred).sum()
    Union = np.logical_or(bool3d_ref, bool3d_pred).sum()
    Dice = 2.0 * Intersection / (Intersection + Union)
    VOE = 1.0 - Intersection / Union
    return Dice, VOE
