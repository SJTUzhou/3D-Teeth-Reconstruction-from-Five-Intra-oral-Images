import os
import sys
import numpy as np
import pcd_mesh_utils as pm_util
import recons_eval_metric as metric
import projection_utils as proj
import functools
import ray
import psutil
import h5py
from emopt5views import EMOpt5Views
from const import *


 

SSM_DIR = r"./ssm/eigValVec/"
REGIS_PARAM_DIR = r"./ssm/cpdGpParams/"

STAGE0_MAT_DIR = r"./stage0-mat/"
DEMO_H5_DIR = r"./demo/h5/"
DEMO_MESH_DIR = r"./demo/mesh/"
LOG_DIR = r"./log/"
NUM_CPUS = psutil.cpu_count(logical=False)
print = functools.partial(print, flush=True)





def run_emopt(emopt):
    # 3d teeth reconstruction by optimization
    print("-"*100)
    print("Start optimization.")

    stage0initMatFile = os.path.join(STAGE0_MAT_DIR, "E-step-result-stage0-init.mat")
    stage0finalMatFile = os.path.join(STAGE0_MAT_DIR, "E-step-result-stage0-final.mat")

    # grid search parallelled by Ray
    print("-"*100)
    print("Start Grid Search.")

    # parallel function supported by Ray
    emopt.searchDefaultRelativePoseParams()
    emopt.gridSearchExtrinsicParams()
    emopt.gridSearchRelativePoseParams()
    
    emopt.expectation_step_5Views(-1, verbose=True)
    # emopt.save_expectation_step_result(stage0initMatFile) # save checkpoint


    maxiter = 20
    stageIter = [10,5,10]
    # stage 0 & 1 optimization

    print("-"*100)
    print("Start Stage 0.")
    stage = 0
    
    # Continue from checkpoint "E-step-result-stage0-init.mat"
    # emopt.load_expectation_step_result(stage0initMatFile, stage) 
    # emopt.expectation_step_5Views(stage, verbose=True)
       
    E_loss = []
    for it in range(stageIter[0]):
        emopt.maximization_step_5Views(stage, step=-1, maxiter=maxiter, verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(stage, verbose=True)
        e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
        print("Sum of expectation step loss: {:.4f}".format(e_loss)) 
        if len(E_loss)>=2 and e_loss>=np.mean(E_loss[-2:]):
            print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
            E_loss.append(e_loss)
            break
        else:
            E_loss.append(e_loss)
    emopt.save_expectation_step_result(stage0finalMatFile) # save checkpoint

    
    skipStage1Flag = False
    print("-"*100)
    print("Start Stage 1.")

    stage = 1
    for it in range(stageIter[1]): 
        emopt.maximization_step_5Views(stage, step=-1, maxiter=maxiter, verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(stage, verbose=True)
        e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        if e_loss >= E_loss[-1]: 
            if it == 0:
                skipStage1Flag = True # first optimization with rowScaleXZ gets worse result compared with optimziaiton without rowScaleXZ
            print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
            break
        else:
            E_loss.append(e_loss)

    print("emopt.rowScaleXZ: ", emopt.rowScaleXZ)
    print("approx tooth scale: ", np.prod(emopt.rowScaleXZ)**(1/3))

    # whether to skip stage1 to avoid extreme deformation
    if skipStage1Flag == True:
        print("Skip Stage 1; Reverse to Stage 0 final result.")
        emopt.rowScaleXZ = np.ones((2,))
        emopt.load_expectation_step_result(stage0finalMatFile, stage=2)
    else:
        print("Accept Stage 1.")
        emopt.anistropicRowScale2ScalesAndTransVecs()    
    emopt.expectation_step_5Views(stage, verbose=True)

    # Stage = 2
    print("-"*100)
    print("Start Stage 2.")
    stage = 2
    E_loss = []
    for it in range(stageIter[2]):
        emopt.maximization_step_5Views(stage, step=2, maxiter=maxiter, verbose=False)
        emopt.maximization_step_5Views(stage, step=3, maxiter=maxiter, verbose=False)
        emopt.maximization_step_5Views(stage=3, step=-1, maxiter=maxiter, verbose=False)
        emopt.maximization_step_5Views(stage, step=1, maxiter=maxiter, verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(stage=3, verbose=True)
        e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        if len(E_loss)>=2 and (e_loss>=np.mean(E_loss[-2:])):
            print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
            break
        else:
            E_loss.append(e_loss)
     
    return emopt



def evaluation(h5File, X_Ref_Upper, X_Ref_Lower):
    '''
    h5file: emopt result saved in h5 format
    X_Ref_Upper, X_Ref_Lower: List of numpy arrays
    '''
    with h5py.File(h5File, 'r') as f:
        grp = f["EMOPT"]
        X_Pred_Upper = grp["UPPER_PRED"][:]
        X_Pred_Lower = grp["LOWER_PRED"][:]

    _X_Ref = X_Ref_Upper + X_Ref_Lower # List concat
    print("Compare prediction shape aligned by similarity registration with ground truth.")
    with_scale = True
    T_Upper = pm_util.computeTransMat(X_Pred_Upper.reshape(-1,3), np.concatenate(X_Ref_Upper), with_scale=with_scale)
    T_Lower = pm_util.computeTransMat(X_Pred_Lower.reshape(-1,3), np.concatenate(X_Ref_Lower), with_scale=with_scale)

    TX_Pred_Upper = np.matmul(X_Pred_Upper, T_Upper[:3,:3]) + T_Upper[3,:3]
    TX_Pred_Lower = np.matmul(X_Pred_Lower, T_Lower[:3,:3]) + T_Lower[3,:3]
    _TX_Pred = np.concatenate([TX_Pred_Upper, TX_Pred_Lower])

    RMSD_T_pred = metric.computeRMSD(_X_Ref, _TX_Pred)
    ASSD_T_pred = metric.computeASSD(_X_Ref, _TX_Pred)
    HD_T_pred = metric.computeHD(_X_Ref, _TX_Pred)
    CD_T_pred = metric.computeChamferDistance(_X_Ref, _TX_Pred)
    print("[RMSD] Root Mean Squared surface Distance (mm): {:.4f}".format(RMSD_T_pred))
    print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(ASSD_T_pred))
    print("[HD] Hausdorff distance (mm): {:.4f}".format(HD_T_pred))
    print("[CD] Chamfer distance (mm^2): {:.4f}".format(CD_T_pred))

    Dice_VOE_lst = [metric.computeDiceAndVOE(_x_ref, _x_pred, pitch=0.2) for _x_ref, _x_pred in zip(_X_Ref, _TX_Pred)]
    avg_Dice, avg_VOE = np.array(Dice_VOE_lst).mean(axis=0)
    print("[DC] Volume Dice Coefficient: {:.4f}".format(avg_Dice))
    print("[VOE] Volumetric Overlap Error: {:.2f} %".format(100.*avg_VOE))


        


@ray.remote
def createAlignedPredMeshes(h5File, X_Ref_Upper, X_Ref_Lower, save_name, meshDir):
    '''
    h5file: emopt result saved in h5 format
    X_Ref_Upper, X_Ref_Lower: List of numpy arrays
    '''
    with h5py.File(h5File, 'r') as f:
        grp = f["EMOPT"]
        X_Pred_Upper = grp["UPPER_PRED"][:]
        X_Pred_Lower = grp["LOWER_PRED"][:]
    with_scale = True
    T_Upper = pm_util.computeTransMat(X_Pred_Upper.reshape(-1,3), np.concatenate(X_Ref_Upper), with_scale=with_scale)
    T_Lower = pm_util.computeTransMat(X_Pred_Lower.reshape(-1,3), np.concatenate(X_Ref_Lower), with_scale=with_scale)

    TX_Pred_Upper = np.matmul(X_Pred_Upper, T_Upper[:3,:3]) + T_Upper[3,:3]
    TX_Pred_Lower = np.matmul(X_Pred_Lower, T_Lower[:3,:3]) + T_Lower[3,:3]

    X_Pred_Upper_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Pred_Upper]
    X_Pred_Lower_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Pred_Lower]
    Pred_Upper_Mesh = pm_util.mergeO3dTriangleMeshes(X_Pred_Upper_Meshes)
    Pred_Lower_Mesh = pm_util.mergeO3dTriangleMeshes(X_Pred_Lower_Meshes)

    TX_Pred_Upper_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in TX_Pred_Upper]
    TX_Pred_Lower_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in TX_Pred_Lower]
    Aligned_Pred_Upper_Mesh = pm_util.mergeO3dTriangleMeshes(TX_Pred_Upper_Meshes)
    Aligned_Pred_Lower_Mesh = pm_util.mergeO3dTriangleMeshes(TX_Pred_Lower_Meshes)

    demoMeshDir = os.path.join(meshDir, "{}/".format(save_name))
    os.makedirs(demoMeshDir, exist_ok=True)
    
    pm_util.exportTriMeshObj(np.asarray(Pred_Upper_Mesh.vertices), np.asarray(Pred_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Pred_Upper_Mesh_Tag={}.obj".format(save_name)))
    pm_util.exportTriMeshObj(np.asarray(Pred_Lower_Mesh.vertices), np.asarray(Pred_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Pred_Lower_Mesh_Tag={}.obj".format(save_name)))
    pm_util.exportTriMeshObj(np.asarray(Aligned_Pred_Upper_Mesh.vertices), np.asarray(Aligned_Pred_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Aligned_Pred_Upper_Mesh_Tag={}.obj".format(save_name)))
    pm_util.exportTriMeshObj(np.asarray(Aligned_Pred_Lower_Mesh.vertices), np.asarray(Aligned_Pred_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Aligned_Pred_Lower_Mesh_Tag={}.obj".format(save_name)))





def main():
    tag = "emopt-demo"
    Mu, SqrtEigVals, Sigma = proj.loadMuEigValSigma(SSM_DIR, numPC=NUM_PC)
    Mu_normals = EMOpt5Views.computePointNormals(Mu)

    transVecStd = 1.1463183505325343 # obtained by SSM
    rotVecStd = 0.13909168140778128 # obtained by SSM
    PoseCovMats = np.load(os.path.join(REGIS_PARAM_DIR, "PoseCovMats.npy")) # Covariance matrix of tooth pose for each tooth, shape=(28,6,6)
    ScaleCovMat = np.load(os.path.join(REGIS_PARAM_DIR, "ScaleCovMat.npy")) # Covariance matrix of scales for each tooth, shape=(28,28)

    str_photo_types = ["upperPhoto","lowerPhoto","leftPhoto","rightPhoto","frontalPhoto"]
    photoTypes = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
    VISIBLE_MASKS = [MASK_UPPER, MASK_LOWER, MASK_LEFT, MASK_RIGHT, MASK_FRONTAL]
    TOOTH_EXIST_MASK = np.ones((28,), np.bool_) # tooth existence of the patient
    
    LogFile = os.path.join(LOG_DIR, "Tag={}.log".format(tag))
    if os.path.exists(LogFile):
        os.remove(LogFile)
    log = open(LogFile, "a", encoding='utf-8')
    sys.stdout = log
        
    
    # TODO
    edgeMasks = [np.empty((800,600)),] * 5
    X_Ref_Upper = [np.empty((1500,3)),] * 14
    X_Ref_Lower = [np.empty((1500,3)),] * 14


    
    emopt = EMOpt5Views(edgeMasks, photoTypes, VISIBLE_MASKS, TOOTH_EXIST_MASK, Mu, Mu_normals, SqrtEigVals, Sigma, PoseCovMats, ScaleCovMat, transVecStd, rotVecStd)
    emopt = run_emopt(emopt)
    demoh5File = os.path.join(DEMO_H5_DIR, "emopt-demo.h5")
    emopt.saveDemo2H5(demoh5File)
    evaluation(demoh5File, X_Ref_Upper, X_Ref_Lower)

    log.close()


if __name__ == "__main__":
    ray.init(num_cpus=4, num_gpus=1)  
    main()
            



    