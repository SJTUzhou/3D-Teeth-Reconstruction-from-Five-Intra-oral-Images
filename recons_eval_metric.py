import numpy as np
import scipy
import trimesh
from trimesh.voxel import creation as tri_creation
from pcd_mesh_utils import surfaceVertices2WatertightO3dMesh


"""Evaluation Metrics for 3D teeth Reconstruction"""

def computeRMSE(X_pred, X_Ref):
    # compute Root Mean Square Error of corresponding points
    pointL2Errors = np.linalg.norm(X_pred - X_Ref, axis=2, ord=2)
    return np.mean(pointL2Errors)

def computeRMSD(X_Pred, X_Ref, return_list=False):
    # compute Root Mean Squared symmetric surface Distance
    RMSDs = []
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
    # compute Average Symmetric Surface Distance
    ASSDs = []
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
    # compute Hausdorff Distance
    HDs = []
    for x_pred, x_ref in zip(X_Pred, X_Ref):
        dist_mat = scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8))
        hd1 = np.max(np.min(dist_mat, axis=0))
        hd2 = np.max(np.min(dist_mat, axis=1))
        HDs.append(np.max([hd1,hd2]))
    if return_list == True:
        return HDs
    return np.mean(HDs)

def computeChamferDistance(X_Pred, X_Ref): 
    CDs = []
    for x_pred, x_ref in zip(X_Pred, X_Ref):
        squaredDistMat = scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8)) ** 2
        CDs.append(np.min(squaredDistMat, axis=0).mean() + np.min(squaredDistMat, axis=1).mean())
    return np.mean(CDs)


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