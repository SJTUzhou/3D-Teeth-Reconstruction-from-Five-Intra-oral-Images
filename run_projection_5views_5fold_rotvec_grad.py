import os
import sys
import numpy as np
import pandas as pd
import csv
import pcd_mesh_utils as pm_util
import recons_eval_metric as metric
import projection_utils as proj
import functools
import ray
import psutil
from projection_utils import PHOTO
from emopt5views_parallel_rotvec_grad import EMOpt5Views


 



# SSM_DIR = r"./data/cpdAlignedData/eigValVec/" 
# CPD_ALIGNED_DIR = r"./data/cpdAlignedData/" 
# PARAM_DIR = r"./data/params/" 
# SRC_DIR = r"./dataWithPhoto/cpdAlignedData/"
# SRC_PARAM_DIR = r"./dataWithPhoto/params/" 

SSM_DIR = r"./data/cpdGpAlignedData/eigValVec/" 
CPD_ALIGNED_DIR = r"./data/cpdGpAlignedData/" 
PARAM_DIR = r"./data/cpdGpParams/" 
SRC_DIR = r"./dataWithPhoto/cpdGpAlignedData/"
SRC_PARAM_DIR = r"./dataWithPhoto/cpdGpParams/"

NUM_PC = 10 # 3
NUM_POINT = 1500
PG_SHAPE = (NUM_POINT, 3)
UPPER_INDICES = [11,12,13,14,15,16,17,21,22,23,24,25,26,27] #不考虑智齿18,28
LOWER_INDICES = [31,32,33,34,35,36,37,41,42,43,44,45,46,47] #不考虑智齿38,48

NAME_IDX_MAP_CSV = r"./dataWithPhoto/nameIndexMapping.csv"

PG_NPY = os.path.join(SRC_PARAM_DIR, "Y_pg.npy")
MASK_NPY = os.path.join(SRC_PARAM_DIR, "X_mask.npy")

MATLAB_PATH = r"./matlab_script"
# FOLD_IDX = 5 # change this index each fold
VERSION = "v21" # version of teeth silouette extraction model
# EDGE_MASK_PATH = r"./dataWithPhoto/learning/fold{}/test/pred-{}/".format(FOLD_IDX,VERSION)
STAGE0_MAT_DIR = os.path.join(MATLAB_PATH, "stage0-mat")
DEMO_H5_DIR = r"./dataWithPhoto/demo/"
DEMO_MESH_DIR = r"./dataWithPhoto/demoMesh/"
NAME_IDX_CSV = pd.read_csv(NAME_IDX_MAP_CSV)
TEMP_DIR = r"./dataWithPhoto/_temp/"
LOG_DIR = r"./dataWithPhoto/log/"
print = functools.partial(print, flush=True)



# TagIDs = [63,47,24,54,10,42,18,90,70,46,71,83,84,56,65,53,39,29,12,73,32,69,5,40,72,43,75,1,28,7,92,91,37, 
#         68,60,81,14,52,9,87,88,80,61,30,41,34,74,25,13,0,16,20,15,19,93,59,49,64,89,6,58,31,86,3,27,35,45,
#         26,36,8,76,55,17,22,82,48,66,62,44,2,38,78,4,67,11,85,50,51,79,57,33,77,94,23,21]

Fold1_TagIDs = [66,62,44,2,38,78,4,67,11,85,50,51,79,57,33,77,94,23,21]
Fold2_TagIDs = [64,89,6,58,31,86,3,27,35,45,26,36,8,76,55,17,22,82,48]
Fold3_TagIDs = [9,87,88,80,61,30,41,34,74,25,13,0,16,20,15,19,93,59,49]
Fold4_TagIDs = [73,32,69,5,40,72,43,75,1,28,7,92,91,37,68,60,81,14,52]
Fold5_TagIDs = [63,47,24,54,10,42,18,90,70,46,71,83,84,56,65,53,39,29,12]

TagIDs = Fold5_TagIDs + Fold4_TagIDs + Fold3_TagIDs + Fold2_TagIDs + Fold1_TagIDs

TagID_Folds = {1:Fold1_TagIDs, 2:Fold2_TagIDs, 3:Fold3_TagIDs, 4:Fold4_TagIDs, 5:Fold5_TagIDs}



# # Run it only once
# utils.saveEigValVec(CPD_ALIGNED_DIR, NumPC2Save=100)

# # Run it only once
# src_invRegistrationParamDF = proj.loadInvRegistrationParams(loadDir=SRC_PARAM_DIR) # 加载待优化点云配准过程中的参数
# Mu, SqrtEigVals, Sigma = proj.loadMuEigValSigma(SSM_DIR, numPC=NUM_PC)
# Y_pg, X_mask, Y_scale, Y_rxyz, Y_txyz, Y_fVec = proj.loadDataSet(src_invRegistrationParamDF, Mu, Sigma, pgShape=PG_SHAPE, srcRootDir=SRC_DIR)
# np.save(PG_NPY, np.array(Y_pg))
# np.save(MASK_NPY, np.array(X_mask))



def meanShapeAlignmentEvaluation(with_scale=False):
    '''使用均值模型与待预测点云真实值配准，评估RMSE'''
    pgs = np.load(PG_NPY)
    masks = np.load(MASK_NPY)
    Mu, SqrtEigVals, Sigma = proj.loadMuEigValSigma(SSM_DIR, numPC=NUM_PC)
    Mu_U, Mu_L = np.split(Mu, 2, axis=0)
    print(Mu.shape)
    rmses = []
    for i in range(len(pgs)):
        pg = pgs[i]
        mask = np.squeeze(masks[i])
        mask_U, mask_L = np.split(mask, 2, axis=0)
        pg_U, pg_L = np.split(pg, 2, axis=0)
        TransMat_U = pm_util.computeTransMatByCorres(Mu_U[mask_U].reshape(-1,3), pg_U[mask_U].reshape(-1,3), with_scale)
        TransMat_L = pm_util.computeTransMatByCorres(Mu_L[mask_L].reshape(-1,3), pg_L[mask_L].reshape(-1,3), with_scale)
        T_Mu_U = np.matmul(Mu_U[mask_U], TransMat_U[:3,:3]) + TransMat_U[3,:3]
        T_Mu_L = np.matmul(Mu_L[mask_L], TransMat_L[:3,:3]) + TransMat_L[3,:3]
        rmse1 = metric.computeRMSE(T_Mu_U, pg_U[mask_U])
        rmse2 = metric.computeRMSE(T_Mu_L, pg_L[mask_L])
        rmses.append(rmse1)
        rmses.append(rmse2)
    print("with_scale: ", with_scale)
    print("Compare ground truth with aligned mean shape.")
    print("[RMSE] mm: {:.4f}".format(np.mean(rmses)))



def run_emopt(TagID, emopt, X_Ref, phase):
    # phase 0: grid search parallelled by Ray
    # phase 1: stage 0,1,2,3 optimization by scipy

    print("TagID: ", TagID)
    # Optimization
    print("-"*100)
    print("Start optimization.")

    stage0initMatFile = os.path.join(STAGE0_MAT_DIR, "E-step-result-stage0-init-{}.mat".format(TagID))
    stage0finalMatFile = os.path.join(STAGE0_MAT_DIR, "TEMP-E-step-result-stage0-final.mat")

    # phase 0: grid search parallelled by Ray
    if phase == 0:

        print("num of observed edge points: ", emopt.M)
        print("ex_rxyz: ", emopt.ex_rxyz)
        print("ex_txyz: ", emopt.ex_txyz)
        print("rela_rxyz: ", emopt.rela_rxyz)
        print("rela_txyz: ", emopt.rela_txyz)
        print("focal length: ", emopt.focLth)
        print("d_pixel: ", emopt.dpix)
        print("u0: {}, v0: {}".format(emopt.u0, emopt.v0))
        print("[RMSE] Root Mean Squared Surface Distance(mm): {:.4f}".format(metric.computeRMSE(emopt.X_trans, X_Ref)))
        print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(metric.computeASSD(emopt.X_trans, X_Ref)))
        print("[HD] Hausdorff distance (mm): {:.4f}".format(metric.computeHD(emopt.X_trans, X_Ref)))


        print("-"*100)
        print("Start Grid Search.")

        # parallel function supported by Ray
        emopt.searchDefaultRelativePoseParams()
        emopt.gridSearchExtrinsicParams()
        emopt.gridSearchRelativePoseParams()

        print("ex_rxyz: ", emopt.ex_rxyz)
        print("ex_txyz: ", emopt.ex_txyz)
        print("rela_rxyz: ", emopt.rela_rxyz)
        print("rela_txyz: ", emopt.rela_txyz)
        print("focal length: ", emopt.focLth)
        print("d_pixel: ", emopt.dpix)
        print("u0: {}, v0: {}".format(emopt.u0, emopt.v0))
        emopt.expectation_step_5Views(-1, verbose=True)

        emopt.save_expectation_step_result_with_XRef(stage0initMatFile, X_Ref)


    elif phase == 1: # phase 1: stage 0 & 1 optimization by Matlab
        

        maxiter = 20
        stageIter = [10,5,10]

        print("-"*100)
        print("Start Stage 0.")
        # Continue from checkpoint "E-step-result-stage0-init-{}.mat"
        stage = 0
        emopt.load_expectation_step_result(stage0initMatFile, stage)
        emopt.expectation_step_5Views(stage, verbose=True)
        print("Root Mean Squared Surface Distance(mm): {:.4f}".format(metric.computeRMSE(emopt.X_trans, X_Ref)))        
        E_loss = []
        for it in range(stageIter[0]):
            emopt.maximization_step_5Views(stage, step=-1, maxiter=maxiter, verbose=False)

            print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
            emopt.expectation_step_5Views(stage, verbose=True)
            e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
            print("Sum of expectation step loss: {:.4f}".format(e_loss))
            print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, metric.computeRMSE(emopt.X_trans, X_Ref)))
            if len(E_loss)>=2 and e_loss>=np.mean(E_loss[-2:]):
                print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
                E_loss.append(e_loss)
                break
            else:
                E_loss.append(e_loss)

        
        emopt.save_expectation_step_result_with_XRef(stage0finalMatFile, X_Ref)

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
            print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, metric.computeRMSE(emopt.X_trans, X_Ref)))
            if e_loss >= E_loss[-1]: # len(E_loss)>=2 and e_loss>=np.mean(E_loss[-2:]):
                # 判断条件1：是否跳过stage1
                if it == 0:
                    skipStage1Flag = True # first optimization with rowScaleXZ gets worse result compared with optimziaiton without rowScaleXZ
                print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
                break
            else:
                E_loss.append(e_loss)


        print("emopt.rowScaleXZ: ", emopt.rowScaleXZ)
        print("approx tooth scale: ", np.prod(emopt.rowScaleXZ)**(1/3))

        # 判断是否跳过stage1
        if skipStage1Flag == True:
            print("Skip Stage 1; Reverse to Stage 0 final result.")
            emopt.rowScaleXZ = np.ones((2,))
            emopt.load_expectation_step_result(stage0finalMatFile, stage=2)
            emopt.expectation_step_5Views(stage, verbose=True)
        else:
            print("Accept Stage 1.")
            emopt.anistropicRowScale2ScalesAndTransVecs()      
        print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, metric.computeRMSE(emopt.X_trans, X_Ref)))

        # Stage = 2
        print("-"*100)
        print("Start Stage 2.")
        stage = 2

        emopt.expectation_step_5Views(stage, verbose=True)

        
        E_loss = []
        for it in range(stageIter[2]):
            # emopt.maximization_step_5Views(stage=2, step=4, maxiter=maxiter, verbose=False)
            emopt.maximization_step_5Views(stage, step=2, maxiter=maxiter, verbose=False)
            emopt.maximization_step_5Views(stage, step=3, maxiter=maxiter, verbose=False)
            emopt.maximization_step_5Views(stage=3, step=-1, maxiter=maxiter, verbose=False)
            emopt.maximization_step_5Views(stage, step=1, maxiter=maxiter, verbose=False)
            print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
            emopt.expectation_step_5Views(stage=3, verbose=True)
            e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
            print("Sum of expectation step loss: {:.4f}".format(e_loss))
            print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, metric.computeRMSE(emopt.X_trans, X_Ref)))
            if len(E_loss)>=2 and (e_loss>=np.mean(E_loss[-2:])):
                print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
                break
            else:
                E_loss.append(e_loss)


        # print("[RMSE] Root Mean Squared Surface Distance(mm): {:.4f}".format(metric.computeRMSE(X_Ref, emopt.X_trans)))
        # print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(metric.computeASSD(X_Ref, emopt.X_trans)))
        # print("[HD] Hausdorff distance (mm): {:.4f}".format(metric.computeHD(X_Ref, emopt.X_trans)))
        
    return emopt



def evaluation(TagID, demoh5File, ret_csv):
    print("TagID: ", TagID)
    patientID = TagID
    X_Mu_Upper, X_Mu_Lower, X_Pred_Upper, X_Pred_Lower, X_Ref_Upper, X_Ref_Lower, rela_R, rela_t = proj.readDemoFromH5(demoh5File, patientID)
    _X_Ref = np.concatenate([X_Ref_Upper,X_Ref_Lower])

    # 相似变换配准后的牙列预测与Ground Truth对比
    print("Compare prediction shape aligned by similarity registration with ground truth.")
    with_scale = True
    T_Upper = pm_util.computeTransMatByCorres(X_Pred_Upper.reshape(-1,3), X_Ref_Upper.reshape(-1,3), with_scale=with_scale)
    T_Lower = pm_util.computeTransMatByCorres(X_Pred_Lower.reshape(-1,3), X_Ref_Lower.reshape(-1,3), with_scale=with_scale)

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

    with open(ret_csv, 'a', newline='') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow([patientID,RMSD_T_pred,ASSD_T_pred,HD_T_pred,CD_T_pred,avg_Dice,avg_VOE])
        f_object.close()
        


@ray.remote
def createAlignedPredMeshes(TagID):
    patientID = TagID
    demoH5File2Load = os.path.join(DEMO_H5_DIR, "demo_TagID={}.h5".format(patientID))
    X_Mu_Upper, X_Mu_Lower, X_Pred_Upper, X_Pred_Lower, X_Ref_Upper, X_Ref_Lower, rela_R, rela_t = proj.readDemoFromH5(demoH5File2Load, patientID)
    with_scale = True
    T_Upper = pm_util.computeTransMatByCorres(X_Pred_Upper.reshape(-1,3), X_Ref_Upper.reshape(-1,3), with_scale=with_scale)
    T_Lower = pm_util.computeTransMatByCorres(X_Pred_Lower.reshape(-1,3), X_Ref_Lower.reshape(-1,3), with_scale=with_scale)

    TX_Pred_Upper = np.matmul(X_Pred_Upper, T_Upper[:3,:3]) + T_Upper[3,:3]
    TX_Pred_Lower = np.matmul(X_Pred_Lower, T_Lower[:3,:3]) + T_Lower[3,:3]

    X_Ref_Upper_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Ref_Upper]
    X_Ref_Lower_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Ref_Lower]
    Ref_Upper_Mesh = pm_util.mergeO3dTriangleMeshes(X_Ref_Upper_Meshes)
    Ref_Lower_Mesh = pm_util.mergeO3dTriangleMeshes(X_Ref_Lower_Meshes)

    X_Pred_Upper_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Pred_Upper]
    X_Pred_Lower_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Pred_Lower]
    Pred_Upper_Mesh = pm_util.mergeO3dTriangleMeshes(X_Pred_Upper_Meshes)
    Pred_Lower_Mesh = pm_util.mergeO3dTriangleMeshes(X_Pred_Lower_Meshes)

    TX_Pred_Upper_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in TX_Pred_Upper]
    TX_Pred_Lower_Meshes = [pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in TX_Pred_Lower]
    Aligned_Pred_Upper_Mesh = pm_util.mergeO3dTriangleMeshes(TX_Pred_Upper_Meshes)
    Aligned_Pred_Lower_Mesh = pm_util.mergeO3dTriangleMeshes(TX_Pred_Lower_Meshes)

    demoMeshDir = os.path.join(DEMO_MESH_DIR, "{}/".format(patientID))
    if not os.path.exists(demoMeshDir):
        os.makedirs(demoMeshDir)
    pm_util.exportTriMeshObj(np.asarray(Ref_Upper_Mesh.vertices), np.asarray(Ref_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Ref_Upper_Mesh_TagID={}.obj".format(patientID)))
    pm_util.exportTriMeshObj(np.asarray(Ref_Lower_Mesh.vertices), np.asarray(Ref_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Ref_Lower_Mesh_TagID={}.obj".format(patientID)))
    pm_util.exportTriMeshObj(np.asarray(Pred_Upper_Mesh.vertices), np.asarray(Pred_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Pred_Upper_Mesh_TagID={}.obj".format(patientID)))
    pm_util.exportTriMeshObj(np.asarray(Pred_Lower_Mesh.vertices), np.asarray(Pred_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Pred_Lower_Mesh_TagID={}.obj".format(patientID)))
    pm_util.exportTriMeshObj(np.asarray(Aligned_Pred_Upper_Mesh.vertices), np.asarray(Aligned_Pred_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Aligned_Pred_Upper_Mesh_TagID={}.obj".format(patientID)))
    pm_util.exportTriMeshObj(np.asarray(Aligned_Pred_Lower_Mesh.vertices), np.asarray(Aligned_Pred_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Aligned_Pred_Lower_Mesh_TagID={}.obj".format(patientID)))
    print("Finish create mesh of TagID: {}".format(patientID))




def main(phase):

    Mu, SqrtEigVals, Sigma = proj.loadMuEigValSigma(SSM_DIR, numPC=NUM_PC)
    Mu_normals = EMOpt5Views.computePointNormals(Mu)

    invRegistrationParamDF = proj.loadInvRegistrationParams(loadDir=PARAM_DIR) # 加载配准过程中的参数
    invParamDF = proj.updateAbsTransVecs(invRegistrationParamDF, Mu) # 将牙列scale转化为每颗牙齿的位移，将每颗牙齿的transVecXYZs在局部坐标系下进行表达

    invScaleMeans, invScaleVars, invRotVecXYZMeans, invRotVecXYZVars, invTransVecXYZMeans, invTransVecXYZVars = proj.getMeanAndVarianceOfInvRegistrationParams(invParamDF)
    scaleStds = np.sqrt(invScaleVars)
    transVecStds = np.sqrt(invTransVecXYZVars)
    rotVecStds = np.sqrt(invRotVecXYZVars)
    print("scaleStd: {:.4f}, transVecStd: {:.4f}, rotVecStd: {:.4f}".format(scaleStds.mean(), transVecStds.mean(), rotVecStds.mean()))

    PoseCovMats = proj.GetPoseCovMats(invParamDF, toothIndices=UPPER_INDICES+LOWER_INDICES) # 每个位置的牙齿的6个变换参数的协方差矩阵,shape=(28,6,6)
    ScaleCovMat = proj.GetScaleCovMat(invParamDF, toothIndices=UPPER_INDICES+LOWER_INDICES) # 牙齿scale的协方差矩阵,shape=(28,28)

    str_photo_types = ["upperPhoto","lowerPhoto","leftPhoto","rightPhoto","frontalPhoto"]
    photoTypes = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
    VISIBLE_MASKS = [proj.MASK_UPPER, proj.MASK_LOWER, proj.MASK_LEFT, proj.MASK_RIGHT, proj.MASK_FRONTAL]
    
    
    
    for ID in TagID_Folds[FOLD_IDX]:
        LogFile = os.path.join(LOG_DIR, "TagID-{}.log".format(ID))
        if os.path.exists(LogFile):
            os.remove(LogFile)
        log = open(LogFile, "a", encoding='utf-8')
        sys.stdout = log
        
        edgeMasks = proj.getEdgeMask(EDGE_MASK_PATH, NAME_IDX_CSV, ID, str_photo_types, resized_width=800, binary=True, activation_thre=0.5)
        Mask = proj.GetMaskByTagId(MASK_NPY, TagId=ID)
        # reference pointcloud
        PG_Ref = proj.GetPGByTagId(PG_NPY, TagId=ID)
        X_Ref = PG_Ref[Mask]
        emopt = EMOpt5Views(edgeMasks, photoTypes, VISIBLE_MASKS, Mask, Mu, Mu_normals, SqrtEigVals, Sigma, PoseCovMats, ScaleCovMat, transVecStds.mean(), rotVecStds.mean())

        emopt = run_emopt(ID, emopt, X_Ref, phase)
        if phase == 1:
            demoh5File = os.path.join(DEMO_H5_DIR, "demo_TagID={}.h5".format(ID))
            emopt.saveDemo2H5(demoh5File, ID, X_Ref)
            ret_csv = os.path.join(TEMP_DIR,r'temp_result_fold{}.csv'.format(FOLD_IDX))
            evaluation(ID, demoh5File, ret_csv)

        log.close()


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        # phase 1 & 2
        FOLD_IDX = int(sys.argv[1]) # command line argument: Fold_Index
        # EDGE_MASK_PATH = r"./dataWithPhoto/learning/fold{}/test/label-revised/".format(FOLD_IDX) # reconstruction with ground-truth segmentation
        EDGE_MASK_PATH = r"./dataWithPhoto/learning/fold{}/test/pred-{}/".format(FOLD_IDX, VERSION)
        main(phase=1)

    else:
        # phase 0
        NUM_CPUS = psutil.cpu_count(logical=False)
        ray.init(num_cpus=NUM_CPUS, num_gpus=1) #ray(多线程)初始化
        for FOLD_IDX in [5,4,3,2,1]:
            # EDGE_MASK_PATH = r"./dataWithPhoto/learning/fold{}/test/label-revised/".format(FOLD_IDX) # reconstruction with ground-truth segmentation
            EDGE_MASK_PATH = r"./dataWithPhoto/learning/fold{}/test/pred-{}/".format(FOLD_IDX, VERSION)
            main(phase=0)
            

        # # create demo triangle meshes
        # NUM_CPUS = psutil.cpu_count(logical=False) 
        # ray.init(num_cpus=NUM_CPUS, num_gpus=1) #ray(多线程)初始化
        # ray.get([createAlignedPredMeshes.remote(TagID) for TagID in TagIDs])

    