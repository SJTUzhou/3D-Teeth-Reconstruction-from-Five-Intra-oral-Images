import os
import sys
import numpy as np
import pandas as pd
import csv
import ray
import psutil
import functools
import time
import emopt_func
import utils
from utils import computeRMSE, computeRMSD, computeASSD, computeHD
import projection_utils as proj
from projection_utils import PHOTO
from emopt5views_cpd import EMOpt5Views


 



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
VERSION = "v21" # version of teeth silouette extraction model

DEMO_H5_DIR = r"./dataWithPhoto/demo/"
DEMO_MESH_DIR = r"./dataWithPhoto/demoMesh/"
TEMP_DIR = r"./dataWithPhoto/"
LOG_DIR = r"./dataWithPhoto/log/"
STAGE0_MAT_DIR = r"./dataWithPhoto//stage0mat/"
for d in [TEMP_DIR, STAGE0_MAT_DIR, DEMO_H5_DIR, DEMO_MESH_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
    
print = functools.partial(print, flush=True)



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
        TransMat_U = utils.computeTransMatByCorres(Mu_U[mask_U].reshape(-1,3), pg_U[mask_U].reshape(-1,3), with_scale)
        TransMat_L = utils.computeTransMatByCorres(Mu_L[mask_L].reshape(-1,3), pg_L[mask_L].reshape(-1,3), with_scale)
        T_Mu_U = np.matmul(Mu_U[mask_U], TransMat_U[:3,:3]) + TransMat_U[3,:3]
        T_Mu_L = np.matmul(Mu_L[mask_L], TransMat_L[:3,:3]) + TransMat_L[3,:3]
        rmse1 = computeRMSE(T_Mu_U, pg_U[mask_U])
        rmse2 = computeRMSE(T_Mu_L, pg_L[mask_L])
        rmses.append(rmse1)
        rmses.append(rmse2)
    print("with_scale: ", with_scale)
    print("Compare ground truth with aligned mean shape.")
    print("[RMSE] mm: {:.4f}".format(np.mean(rmses)))



def run_emopt(TagID, emopt, X_Ref):
    # phase 0: grid search parallelled by Ray
    # phase 1: stage 0,1,2,3 optimization by scipy
    
    def __print_camera_params(emopt):
        print("num of observed edge points: ", emopt.M)
        print("ex_rxyz: ", emopt.ex_rxyz)
        print("ex_txyz: ", emopt.ex_txyz)
        print("rela_rxyz: ", emopt.rela_rxyz)
        print("rela_txyz: ", emopt.rela_txyz)
        print("focal length: ", emopt.focLth)
        print("d_pixel: ", emopt.dpix)
        print("u0: {}, v0: {}".format(emopt.u0, emopt.v0))
        
    def __print_metrics(emopt):
        print("[RMSE] Root Mean Squared Surface Distance(mm): {:.4f}".format(computeRMSE(emopt.X_trans, X_Ref)))
        print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(computeASSD(emopt.X_trans, X_Ref)))
        print("[HD] Hausdorff distance (mm): {:.4f}".format(computeHD(emopt.X_trans, X_Ref)))


    print("TagID: ", TagID)
    stage0MatFile = os.path.join(STAGE0_MAT_DIR, "E-step-result-stage0-{}.mat".format(TagID))

    # Optimization

    print("-"*100)
    print("Start optimization.")
    __print_camera_params(emopt)
    # __print_metrics(emopt)


    # phase 0: grid search
    print("-"*100)
    print("Start Grid Search.")

    emopt.searchDefaultRelativePoseParams()
    emopt.gridSearchExtrinsicParams()
    emopt.gridSearchRelativePoseParams()

    
    maxiter = [20,10,10]
    stageIter = [10,5,15]

    print("-"*100)
    print("Start Stage 0.")
    # Continue from checkpoint
    stage = 0
    
    # __print_camera_params(emopt)
    emopt.expectation_step_5Views(stage, verbose=True)
    emopt.save_expectation_step_result_with_XRef(stage0MatFile, X_Ref)

    # stage0MatFile = r".\matlab_script\stage0-mat\E-step-result-stage0-init-{}.mat".format(TagID) # old stage0Mat file
    # emopt.load_expectation_step_result(stage0MatFile, stage)
    # emopt.expectation_step_5Views(stage, verbose=True)
    # print("Root Mean Squared Surface Distance(mm): {:.4f}".format(computeRMSE(emopt.X_trans, X_Ref)))        
    
    E_loss = []
    for it in range(stageIter[0]):
        emopt.maximization_step_5Views(stage, step=-1, maxiter=maxiter[0], verbose=False)

        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(stage, verbose=True)
        e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, computeRMSE(emopt.X_trans, X_Ref)))
        if len(E_loss)>=2 and e_loss>=np.mean(E_loss[-2:]):
            print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
            E_loss.append(e_loss)
            break
        else:
            E_loss.append(e_loss)


    print("-"*100)
    print("Start Stage 1.")
    skipStage1Flag = False
    
    stage = 1
    for it in range(stageIter[1]): 
        emopt.maximization_step_5Views(stage, step=-1, maxiter=maxiter[1], verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(stage, verbose=True)
        e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, computeRMSE(emopt.X_trans, X_Ref)))
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
        emopt.load_expectation_step_result(stage0MatFile, stage=2)
        emopt.expectation_step_5Views(stage=1, verbose=True)
    else:
        print("Accept Stage 1.")
        emopt.anistropicRowScale2ScalesAndTransVecs()      
    print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, computeRMSE(emopt.X_trans, X_Ref)))



    # Stage = 2 & 3
    print("-"*100)
    print("Start Stage 2 & 3.")
    stage = 2

    
    E_loss = []
    for it in range(stageIter[2]):
        # emopt.maximization_step_5Views(stage=2, step=4, maxiter=maxiter[2], verbose=False)
        emopt.maximization_step_5Views(stage, step=2, maxiter=maxiter[2], verbose=False)
        emopt.maximization_step_5Views(stage, step=3, maxiter=maxiter[2], verbose=False)
        emopt.maximization_step_5Views(stage=3, step=-1, maxiter=maxiter[2], verbose=False)
        emopt.maximization_step_5Views(stage, step=1, maxiter=maxiter[2], verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(stage=3, verbose=True)
        e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, computeRMSE(emopt.X_trans, X_Ref)))
        if len(E_loss)>=2 and (e_loss>=np.mean(E_loss[-2:])):
            print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
            break
        else:
            E_loss.append(e_loss)


    # __print_camera_params(emopt)
    # __print_metrics(emopt)
    
    return emopt



def evaluation(TagID):
    
    def __get_metrics(X1, X2):
        _rmse = computeRMSE(X1, X2)
        _rmsd = computeRMSD(X1, X2)
        _assd = computeASSD(X1, X2)
        _hd = computeHD(X1, X2)
        print("[RMSE] Root Mean Squared Error of corresponding points (mm): {:.4f}".format(_rmse))
        print("[RMSD] Root Mean Squared surface Distance (mm): {:.4f}".format(_rmsd))
        print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(_assd))
        print("[HD] Hausdorff distance (mm): {:.4f}".format(_hd))
        return _rmse, _rmsd, _assd, _hd
        
    print("TagID: ", TagID)
    patientID = TagID
    demoH5File2Load = os.path.join(DEMO_H5_DIR, "demo_TagID={}.h5".format(patientID))
    X_Mu_Upper, X_Mu_Lower, X_Pred_Upper, X_Pred_Lower, X_Ref_Upper, X_Ref_Lower, rela_R, rela_t = proj.readDemoFromH5(demoH5File2Load, patientID)
    _X_Ref = np.concatenate([X_Ref_Upper,X_Ref_Lower])
    _X_Mu = np.concatenate([X_Mu_Upper,X_Mu_Lower])
    _X_Pred = np.concatenate([X_Pred_Upper,X_Pred_Lower])

    # 牙列均值与Ground Truth对比
    print("Compare Mean shape with ground truth.")
    __get_metrics(_X_Ref, _X_Mu)

    # 牙列预测与Ground Truth对比
    print("Compare prediction shape with ground truth.")
    RMSE_pred, RMSD_pred, ASSD_pred, HD_pred = __get_metrics(_X_Ref, _X_Pred)

    # # 上下牙列咬合时的相对位置的预测
    # X_assem = np.concatenate([X_Pred_Upper, np.matmul(X_Pred_Lower, rela_R)+rela_t], axis=0)
    # utils.showPointCloud(X_assem.reshape(-1,3), "上下牙列咬合时的相对位置关系的预测")

    # 相似变换配准后的牙列预测与Ground Truth对比
    print("Compare prediction shape aligned by similarity registration with ground truth.")
    with_scale = True
    T_Upper = utils.computeTransMatByCorres(X_Pred_Upper.reshape(-1,3), X_Ref_Upper.reshape(-1,3), with_scale=with_scale)
    T_Lower = utils.computeTransMatByCorres(X_Pred_Lower.reshape(-1,3), X_Ref_Lower.reshape(-1,3), with_scale=with_scale)

    TX_Pred_Upper = np.matmul(X_Pred_Upper, T_Upper[:3,:3]) + T_Upper[3,:3]
    TX_Pred_Lower = np.matmul(X_Pred_Lower, T_Lower[:3,:3]) + T_Lower[3,:3]
    _TX_Pred = np.concatenate([TX_Pred_Upper, TX_Pred_Lower])
    
    RMSE_T_pred, RMSD_T_pred, ASSD_T_pred, HD_T_pred = __get_metrics(_X_Ref, _TX_Pred)


    Dice_VOE_lst = [utils.computeDiceAndVOE(_x_ref, _x_pred, pitch=0.2) for _x_ref, _x_pred in zip(_X_Ref, _TX_Pred)]
    avg_Dice, avg_VOE = np.array(Dice_VOE_lst).mean(axis=0)
    print("[DC] Volume Dice Coefficient: {:.4f}".format(avg_Dice))
    print("[VOE] Volumetric Overlap Error: {:.2f} %".format(100.*avg_VOE))

    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open(os.path.join(TEMP_DIR, r'temp_result.csv'), 'a', newline='') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow([patientID,RMSE_pred,RMSD_pred,ASSD_pred,HD_pred,RMSE_T_pred,RMSD_T_pred,ASSD_T_pred,HD_T_pred,avg_Dice,avg_VOE])
        f_object.close()



@ray.remote
def createAlignedPredMeshes(TagID):
    patientID = TagID
    demoH5File2Load = os.path.join(DEMO_H5_DIR, "demo_TagID={}.h5".format(patientID))
    X_Mu_Upper, X_Mu_Lower, X_Pred_Upper, X_Pred_Lower, X_Ref_Upper, X_Ref_Lower, rela_R, rela_t = proj.readDemoFromH5(demoH5File2Load, patientID)
    with_scale = True
    T_Upper = utils.computeTransMatByCorres(X_Pred_Upper.reshape(-1,3), X_Ref_Upper.reshape(-1,3), with_scale=with_scale)
    T_Lower = utils.computeTransMatByCorres(X_Pred_Lower.reshape(-1,3), X_Ref_Lower.reshape(-1,3), with_scale=with_scale)

    TX_Pred_Upper = np.matmul(X_Pred_Upper, T_Upper[:3,:3]) + T_Upper[3,:3]
    TX_Pred_Lower = np.matmul(X_Pred_Lower, T_Lower[:3,:3]) + T_Lower[3,:3]

    X_Ref_Upper_Meshes = [utils.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Ref_Upper]
    X_Ref_Lower_Meshes = [utils.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Ref_Lower]
    Ref_Upper_Mesh = utils.mergeO3dTriangleMeshes(X_Ref_Upper_Meshes)
    Ref_Lower_Mesh = utils.mergeO3dTriangleMeshes(X_Ref_Lower_Meshes)

    X_Pred_Upper_Meshes = [utils.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Pred_Upper]
    X_Pred_Lower_Meshes = [utils.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Pred_Lower]
    Pred_Upper_Mesh = utils.mergeO3dTriangleMeshes(X_Pred_Upper_Meshes)
    Pred_Lower_Mesh = utils.mergeO3dTriangleMeshes(X_Pred_Lower_Meshes)

    TX_Pred_Upper_Meshes = [utils.surfaceVertices2WatertightO3dMesh(pg) for pg in TX_Pred_Upper]
    TX_Pred_Lower_Meshes = [utils.surfaceVertices2WatertightO3dMesh(pg) for pg in TX_Pred_Lower]
    Aligned_Pred_Upper_Mesh = utils.mergeO3dTriangleMeshes(TX_Pred_Upper_Meshes)
    Aligned_Pred_Lower_Mesh = utils.mergeO3dTriangleMeshes(TX_Pred_Lower_Meshes)

    demoMeshDir = os.path.join(DEMO_MESH_DIR, "{}/".format(patientID))
    if not os.path.exists(demoMeshDir):
        os.makedirs(demoMeshDir)
    utils.exportTriMeshObj(np.asarray(Ref_Upper_Mesh.vertices), np.asarray(Ref_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Ref_Upper_Mesh_TagID={}.obj".format(patientID)))
    utils.exportTriMeshObj(np.asarray(Ref_Lower_Mesh.vertices), np.asarray(Ref_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Ref_Lower_Mesh_TagID={}.obj".format(patientID)))
    utils.exportTriMeshObj(np.asarray(Pred_Upper_Mesh.vertices), np.asarray(Pred_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Pred_Upper_Mesh_TagID={}.obj".format(patientID)))
    utils.exportTriMeshObj(np.asarray(Pred_Lower_Mesh.vertices), np.asarray(Pred_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Pred_Lower_Mesh_TagID={}.obj".format(patientID)))
    utils.exportTriMeshObj(np.asarray(Aligned_Pred_Upper_Mesh.vertices), np.asarray(Aligned_Pred_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Aligned_Pred_Upper_Mesh_TagID={}.obj".format(patientID)))
    utils.exportTriMeshObj(np.asarray(Aligned_Pred_Lower_Mesh.vertices), np.asarray(Aligned_Pred_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Aligned_Pred_Lower_Mesh_TagID={}.obj".format(patientID)))
    print("Finish create mesh of TagID: {}".format(patientID))



def run_experiments():
    __console = sys.stdout
    tic_load = time.time()
    
    # 顺序相互对应
    str_photo_types = ["upperPhoto","lowerPhoto","leftPhoto","rightPhoto","frontalPhoto"]
    
    # 初始化
    photoTypes = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
    VISIBLE_MASKS = [proj.MASK_UPPER, proj.MASK_LOWER, proj.MASK_LEFT, proj.MASK_RIGHT, proj.MASK_FRONTAL]
    Mu, SqrtEigVals, Sigma = proj.loadMuEigValSigma(SSM_DIR, numPC=NUM_PC)
    Mu_normals = emopt_func.computeGroupedPointNormals(Mu)

    invRegistrationParamDF = proj.loadInvRegistrationParams(loadDir=PARAM_DIR) # 加载配准过程中的参数
    invParamDF = proj.updateAbsTransVecs(invRegistrationParamDF, Mu) # 将牙列scale转化为每颗牙齿的位移，将每颗牙齿的transVecXYZs在局部坐标系下进行表达

    invScaleMeans, invScaleVars, invRotVecXYZMeans, invRotVecXYZVars, invTransVecXYZMeans, invTransVecXYZVars = proj.getMeanAndVarianceOfInvRegistrationParams(invParamDF)
    scaleStds = np.sqrt(invScaleVars)
    transVecStds = np.sqrt(invTransVecXYZVars)
    rotVecStds = np.sqrt(invRotVecXYZVars)
    print("scaleStd: {:.4f}, transVecStd: {:.4f}, rotVecStd: {:.4f}".format(scaleStds.mean(), transVecStds.mean(), rotVecStds.mean()))

    PoseCovMats = proj.GetPoseCovMats(invParamDF, toothIndices=UPPER_INDICES+LOWER_INDICES) # 每个位置的牙齿的6个变换参数的协方差矩阵,shape=(28,6,6)
    ScaleCovMat = proj.GetScaleCovMat(invParamDF, toothIndices=UPPER_INDICES+LOWER_INDICES) # 牙齿scale的协方差矩阵,shape=(28,28)
    name_idx_df = pd.read_csv(NAME_IDX_MAP_CSV)
    
    toc_load = time.time()
    init_time = toc_load-tic_load
    print(f"Initialization time: {init_time:.2f} s.")
    
    tictocs = []
    
    for FOLD_IDX in [4,]:
        EDGE_MASK_PATH = r"./dataWithPhoto/learning/fold{}/test/pred-{}/".format(FOLD_IDX, VERSION)
        
        for TagID in TagID_Folds[FOLD_IDX][-8:]:
            tic = time.time()
            
            LogFile = os.path.join(LOG_DIR, "TagID-{}.log".format(TagID))
            # Log file
            if os.path.exists(LogFile):
                os.remove(LogFile)
            log = open(LogFile, "a", encoding='utf-8')
            sys.stdout = log
            
            edgeMasks = proj.getEdgeMask(EDGE_MASK_PATH, name_idx_df, TagID, str_photo_types, resized_width=800, binary=True, activation_thre=0.5)
            Mask = proj.GetMaskByTagId(MASK_NPY, TagId=TagID)
            PG_Ref = proj.GetPGByTagId(PG_NPY, TagId=TagID)
            X_Ref = PG_Ref[Mask]

            emopt = EMOpt5Views(edgeMasks, photoTypes, VISIBLE_MASKS, Mask, Mu,\
                Mu_normals, SqrtEigVals, Sigma, PoseCovMats, ScaleCovMat, transVecStds.mean(), rotVecStds.mean())
            
            emopt = run_emopt(TagID, emopt, X_Ref)
        
            toc = time.time()

            print(f"Grid search and optimization time: {toc-tic:.2f} s.")
            tictocs.append(toc-tic)
            
            
            # print("-"*100)
            # print("Evaluation.")
            # invRotVecXYZVars = invRotVecXYZVars.reshape(-1,3)
            # invTransVecXYZVars = invTransVecXYZVars.reshape(-1,3)
            # print("standard transVecXYZs:")
            # print((emopt.transVecXYZs - emopt.meanTransVecXYZs) / np.sqrt(invTransVecXYZVars[Mask]))
            # print("standard rotVecXYZs:")
            # print((emopt.rotVecXYZs - emopt.meanRotVecXYZs) / np.sqrt(invRotVecXYZVars[Mask]))
            # print("scales:")
            # print(emopt.scales)
            # print("feature vectors:")
            # print(emopt.featureVec)
            
            # # 不考虑第二磨牙
            # withoutSecondMolarMask = np.tile(np.array([1,1,1,1,1,1,0],dtype=np.bool_),(4,))
            # print("Without Second Molar, Root Mean Squared Surface Distance(mm): {:.4f}".format(computeRMSE(emopt.X_trans[withoutSecondMolarMask[Mask]], X_Ref[withoutSecondMolarMask[Mask]])))

            # Save Demo Result
            demoH5File = os.path.join(DEMO_H5_DIR, "demo_TagID={}.h5".format(TagID))
            emopt.saveDemo2H5(demoH5File, TagID, X_Ref)
            
            evaluation(TagID)
            
            log.close()
        
        sys.stdout = __console
        print(f"Average execution time: {init_time + np.mean(tictocs):.2f} s.")
    
    sys.stdout = __console
    print(f"Average execution time: {init_time + np.mean(tictocs):.2f} s.")





if __name__ == "__main__":
    ray.init(num_cpus=4)
    run_experiments()
    
    # for tagId in TagIDs:
    #     evaluation(tagId)

    # create demo triangle meshes
    # NUM_CPUS = psutil.cpu_count(logical=False) 
    # ray.init(num_cpus=NUM_CPUS, num_gpus=1) #ray(多线程)初始化
    # ray.get([utils.createAlignedPredMeshes.remote(TagID, DEMO_H5_DIR, DEMO_MESH_DIR) for TagID in TagIDs])
