import os
import sys
import numpy as np
import pandas as pd
import csv
import open3d as o3d
from matplotlib import pyplot as plt
import scipy
import scipy.io
import utils
from utils import computeRMSE, computeASSD, computeHD
import projection_utils as proj
import functools
import matlab.engine
import ray
import psutil
from emopt5views_matlab_parallel_euler import EMOpt5Views, PHOTO




ENGINE = matlab.engine.connect_matlab()

SSM_DIR = r"./data/cpdAlignedData/eigValVec/"
CPD_ALIGNED_DIR = r"./data/cpdAlignedData/"
NAME_IDX_MAP_CSV = r"./dataWithPhoto/nameIndexMapping.csv"
EDGE_MASK_PATH = r"./dataWithPhoto/normal_mask/"
RGB_EDGE_MASK_PATH = r"./dataWithPhoto/normal_mask/RGB/"
NUM_PC = 3
PG_NPY = os.path.join("res-2D-3D", "Y_pg.npy")
MASK_NPY = os.path.join("res-2D-3D", "X_mask.npy")
PARAM_DIR = r"./data/params/"
UPPER_INDICES = [11,12,13,14,15,16,17,21,22,23,24,25,26,27] #不考虑智齿18,28
LOWER_INDICES = [31,32,33,34,35,36,37,41,42,43,44,45,46,47] #不考虑智齿38,48
MATLAB_PATH = r"./matlab_script"
STAGE0_MAT_DIR = os.path.join(MATLAB_PATH, "stage0-mat")

NAME_IDX_CSV = pd.read_csv(NAME_IDX_MAP_CSV)

LOG_DIR = r"./dataWithPhoto/log/"
print = functools.partial(print, flush=True)



TagIDs = [63,47,24,54,10,42,18,90,70,46,71,83,84,56,65,53,39,29,12,73,32,69,5,40,72,43,75,1,28,7,92,91,37, 
        68,60,81,14,52,9,87,88,80,61,30,41,34,74,25,13,0,16,20,15,19,93,59,49,64,89,6,58,31,86,3,27,35,45,
        26,36,8,76,55,17,22,82,48,66,62,44,2,38,78,4,67,11,85,50,51,79,57,33,77,94,23,21]








def run_emopt(TagID):
    print("TagID: ", TagID)

    # 顺序相互对应
    photo_types = ["upperPhoto","lowerPhoto","leftPhoto","rightPhoto","frontalPhoto"]
    edgeMasks = proj.getEdgeMask(EDGE_MASK_PATH, NAME_IDX_CSV, TagID, photo_types, resized_width=800)
    # proj.visualizeEdgeMasks(edgeMasks, photo_types)

    Mu, SqrtEigVals, Sigma = proj.loadMuEigValSigma(SSM_DIR, numPC=NUM_PC)

    invRegistrationParamDF = proj.loadInvRegistrationParams(loadDir=PARAM_DIR) # 加载配准过程中的参数
    invParamDF = proj.updateAbsTransVecs(invRegistrationParamDF, Mu) # 将牙列scale转化为每颗牙齿的位移，将每颗牙齿的transVecXYZs在局部坐标系下进行表达

    invScaleMeans, invScaleVars, invRotAngleXYZMeans, invRotAngleXYZVars, invTransVecXYZMeans, invTransVecXYZVars = proj.getMeanAndVarianceOfInvRegistrationParams(invParamDF)
    scaleStds = np.sqrt(invScaleVars)
    transVecStds = np.sqrt(invTransVecXYZVars)
    rotAngleStds = np.sqrt(invRotAngleXYZVars)
    print("scaleStd: {:.4f}, transVecStd: {:.4f}, rotAngleStd: {:.4f}".format(scaleStds.mean(), transVecStds.mean(), rotAngleStds.mean()))

    PoseCovMats = proj.GetPoseCovMats(invParamDF, toothIndices=UPPER_INDICES+LOWER_INDICES) # 每个位置的牙齿的6个变换参数的协方差矩阵,shape=(28,6,6)
    ScaleCovMat = proj.GetScaleCovMat(invParamDF, toothIndices=UPPER_INDICES+LOWER_INDICES) # 牙齿scale的协方差矩阵,shape=(28,28)
    
    Mask = proj.GetMaskByTagId(MASK_NPY, TagId=TagID)
    # reference pointcloud
    PG_Ref = proj.GetPGByTagId(PG_NPY, TagId=TagID)
    X_Ref = PG_Ref[Mask]

    # Optimization

    print("-"*100)
    print("Start optimization.")
    # 初始化
    photoTypes = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
    VISIBLE_MASKS = [proj.MASK_UPPER, proj.MASK_LOWER, proj.MASK_LEFT, proj.MASK_RIGHT, proj.MASK_FRONTAL]
    emopt = EMOpt5Views(edgeMasks, photoTypes, VISIBLE_MASKS, Mask, Mu, SqrtEigVals, Sigma, PoseCovMats, ScaleCovMat, transVecStds.mean(), rotAngleStds.mean())

    E_step_result_file = os.path.join(MATLAB_PATH, "E-step-result.mat")
    M_step_result_file = os.path.join(MATLAB_PATH, "M-step-result.mat")

    print("num of observed edge points: ", emopt.M)
    print("ex_rxyz: ", emopt.ex_rxyz)
    print("ex_txyz: ", emopt.ex_txyz)
    print("rela_rxyz: ", emopt.rela_rxyz)
    print("rela_txyz: ", emopt.rela_txyz)
    print("focal length: ", emopt.focLth)
    print("d_pixel: ", emopt.dpix)
    print("u0: {}, v0: {}".format(emopt.u0, emopt.v0))
    print("[RMSE] Root Mean Squared Surface Distance(mm): {:.4f}".format(computeRMSE(emopt.X_trans, X_Ref)))
    print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(computeASSD(emopt.X_trans, X_Ref)))
    print("[HD] Hausdorff distance (mm): {:.4f}".format(computeHD(emopt.X_trans, X_Ref)))


    print("-"*100)
    print("Start Grid Search.")

    # emopt.searchDefaultRelativePoseParams()
    # emopt.gridSearchExtrinsicParams()
    # emopt.gridSearchRelativePoseParams()

    emopt.expectation_step_5Views(verbose=True)


    stage0initMatFile = os.path.join(STAGE0_MAT_DIR, "E-step-result-stage0-init-{}.mat".format(TagID))
    # proj.saveTempEmOptParamsWithXRef(stage0initMatFile, emopt, X_Ref)

    # canvasShape = (720,960)
    # for photoType in photoTypes:
    #     emopt.showEdgeMaskPredictionWithGroundTruth(photoType, canvasShape, dilate=True)


    print("-"*100)
    print("Start Stage 0.")
    # Continue from checkpoint "E-step-result-stage0-init-{}.mat"

    stage = 0
    ENGINE.addpath(MATLAB_PATH)
    ENGINE.run_MStep(stage, 500, stage0initMatFile, M_step_result_file, nargout=0)
    emopt.load_maximization_step_result(M_step_result_file, stage)

    # emopt.load_expectation_step_result(stage0initMatFile, stage)
    emopt.expectation_step_5Views(verbose=True)
    print("Root Mean Squared Surface Distance(mm): {:.4f}".format(computeRMSE(emopt.X_trans, X_Ref)))

    stage = 0
    maxFuncEval = 500
    E_loss = []
    for it in range(15):
        emopt.maximization_step_5Views_by_Matlab(MATLAB_PATH, ENGINE, stage, maxFuncEval)
        # emopt.maximization_step_5Views(stage, step=-1, rhobeg=1.0, maxiter=maxFuncEval, verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(verbose=True)
        e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, computeRMSE(emopt.X_trans, X_Ref)))
        if len(E_loss)>=2 and e_loss>=np.mean(E_loss[-2:]):
            print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
            E_loss.append(e_loss)
            break
        else:
            E_loss.append(e_loss)


    stage0finalMatFile = os.path.join(STAGE0_MAT_DIR, "E-step-result-stage0-final-{}.mat".format(TagID))
    proj.saveTempEmOptParamsWithXRef(stage0finalMatFile, emopt, X_Ref)

    skipStage1Flag = False
    # # 判断条件1：是否跳过stage1
    # if E_loss[-1] > 500:
    #     skipStage1Flag = True

    print("-"*100)
    print("Start Stage 1.")

    stage = 1
    maxFuncEval = 800
    for it in range(5):
        emopt.maximization_step_5Views_by_Matlab(MATLAB_PATH, ENGINE, stage, maxFuncEval)    
        # emopt.maximization_step_5Views(stage, step=-1, rhobeg=1.0, maxiter=maxFuncEval, verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(verbose=True)
        e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, computeRMSE(emopt.X_trans, X_Ref)))
        if e_loss >= E_loss[-1]: # len(E_loss)>=2 and e_loss>=np.mean(E_loss[-2:]):
            # 判断条件2：是否跳过stage1
            if it == 0:
                skipStage1Flag = True # first optimization with rowScaleXZ gets worse result compared with optimziaiton without rowScaleXZ
            print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
            break
        else:
            E_loss.append(e_loss)


    print("emopt.rowScaleXZ: ", emopt.rowScaleXZ)
    print("approx tooth scale: ", np.prod(emopt.rowScaleXZ)**(1/3))

    # # 判断条件3：是否跳过stage1
    # _rowScaleX, _rowScaleZ = emopt.rowScaleXZ
    # _rowScaleDiff = np.abs(_rowScaleX - _rowScaleZ)
    # _minDiverge = np.abs(1. - emopt.rowScaleXZ).min()
    # if _minDiverge > 0.02 and _rowScaleDiff < _minDiverge:
    #     skipStage1Flag = True

    # 判断是否跳过stage1
    if skipStage1Flag == True:
        print("Skip Stage 1; Reverse to Stage 0 final result.")
        emopt.rowScaleXZ = np.ones((2,))
        ENGINE.run_MStep(0, 500, stage0finalMatFile, M_step_result_file, nargout=0)
        emopt.load_maximization_step_result(M_step_result_file, 2)
        # emopt.load_expectation_step_result(stage0finalMatFile, stage=2)
        emopt.expectation_step_5Views(verbose=True)
    else:
        print("Accept Stage 1.")
        emopt.anistropicRowScale2ScalesAndTransVecs()      
    print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, computeRMSE(emopt.X_trans, X_Ref)))



    # Stage = 2
    print("-"*100)
    print("Start Stage 2.")
    stage = 2
    E_loss = []
    for it in range(10):
        emopt.maximization_step_5Views(stage, step=2, rhobeg=0.1, maxiter=1000, verbose=False)
        emopt.maximization_step_5Views(stage, step=3, rhobeg=0.1, maxiter=1000, verbose=False)
        emopt.maximization_step_5Views(stage=3, step=-1, rhobeg=0.1, maxiter=1000, verbose=False)
        emopt.maximization_step_5Views(stage, step=1, rhobeg=0.1, maxiter=1000, verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(verbose=True)
        e_loss = np.sum(emopt.weightViews * emopt.loss_expectation_step)
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        print("iteration: {}, real Root Mean Squared Surface Distance(mm): {:.4f}".format(it, computeRMSE(emopt.X_trans, X_Ref)))
        if len(E_loss)>=2 and (e_loss>=np.mean(E_loss[-2:])):
            print("Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(E_loss[-2],E_loss[-1],e_loss))
            break
        else:
            E_loss.append(e_loss)


    print("[RMSE] Root Mean Squared Surface Distance(mm): {:.4f}".format(computeRMSE(X_Ref, emopt.X_trans)))
    print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(computeASSD(X_Ref, emopt.X_trans)))
    print("[HD] Hausdorff distance (mm): {:.4f}".format(computeHD(X_Ref, emopt.X_trans)))




    # canvasShape = (720,960)
    # for photoType in photoTypes:
    #     emopt.showEdgeMaskPredictionWithGroundTruth(photoType, canvasShape, dilate=True)


    print("-"*100)
    print("Evaluation.")

    invRotAngleXYZVars = invRotAngleXYZVars.reshape(-1,3)
    invTransVecXYZVars = invTransVecXYZVars.reshape(-1,3)
    print("standard transVecXYZs:")
    print((emopt.transVecXYZs - emopt.meanTransVecXYZs) / np.sqrt(invTransVecXYZVars[Mask]))
    print("standard rotAngleXYZs:")
    print((emopt.rotAngleXYZs - emopt.meanRotAngleXYZs) / np.sqrt(invRotAngleXYZVars[Mask]))
    print("scales:")
    print(emopt.scales)



    # 不考虑第二磨牙
    withoutSecondMolarMask = np.tile(np.array([1,1,1,1,1,1,0],dtype=np.bool_),(4,))
    print("Without Second Molar, Root Mean Squared Surface Distance(mm): {:.4f}".format(computeRMSE(emopt.X_trans[withoutSecondMolarMask[Mask]], X_Ref[withoutSecondMolarMask[Mask]])))



    # # 上下牙列不同的相对位移对应的位置可视化
    # Xtrans = np.concatenate([emopt.X_trans[:emopt.numUpperTooth], np.matmul(emopt.X_trans[emopt.numUpperTooth:],emopt.rela_R)+emopt.rela_txyz], axis=0)
    # utils.showPointCloud(Xtrans.reshape(-1,3), "")


    # Save Demo Result

    demoH5File = r"./dataWithPhoto/demo/demo_TagID={}.h5".format(TagID)
    proj.saveDemo2H5(demoH5File, emopt, TagID, X_Ref)






def evaluation(TagID):
    print("TagID: ", TagID)
    patientID = TagID
    demoH5File2Load = r"./dataWithPhoto/demo/demo_TagID={}.h5".format(patientID)
    X_Mu_Upper, X_Mu_Lower, X_Pred_Upper, X_Pred_Lower, X_Ref_Upper, X_Ref_Lower, rela_R, rela_t = proj.readDemoFromH5(demoH5File2Load, patientID)
    _X_Ref = np.concatenate([X_Ref_Upper,X_Ref_Lower])
    _X_Mu = np.concatenate([X_Mu_Upper,X_Mu_Lower])
    _X_Pred = np.concatenate([X_Pred_Upper,X_Pred_Lower])

    # 牙列均值与Ground Truth对比
    print("Compare Mean shape with ground truth.")
    # proj.showPredPointClouds(y=X_Ref_Upper.reshape(-1,3),py=X_Mu_Upper.reshape(-1,3),mode=2)
    # proj.showPredPointClouds(y=X_Ref_Lower.reshape(-1,3),py=X_Mu_Lower.reshape(-1,3),mode=3)
    print("[RMSE] Root Mean Squared Surface Distance(mm): {:.4f}".format(computeRMSE(_X_Ref, _X_Mu)))
    print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(computeASSD(_X_Ref, _X_Mu)))
    print("[HD] Hausdorff distance (mm): {:.4f}".format(computeHD(_X_Ref, _X_Mu)))


    # 牙列预测与Ground Truth对比
    print("Compare prediction shape with ground truth.")
    # proj.showPredPointClouds(y=X_Ref_Upper.reshape(-1,3),py=X_Pred_Upper.reshape(-1,3),mode=2)
    # proj.showPredPointClouds(y=X_Ref_Lower.reshape(-1,3),py=X_Pred_Lower.reshape(-1,3),mode=3)
    RMSE_pred = computeRMSE(_X_Ref, _X_Pred)
    ASSD_pred = computeASSD(_X_Ref, _X_Pred)
    HD_pred = computeHD(_X_Ref, _X_Pred)
    print("[RMSE] Root Mean Squared Surface Distance(mm): {:.4f}".format(RMSE_pred))
    print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(ASSD_pred))
    print("[HD] Hausdorff distance (mm): {:.4f}".format(HD_pred))

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

    RMSE_T_pred = computeRMSE(_X_Ref, _TX_Pred)
    ASSD_T_pred = computeASSD(_X_Ref, _TX_Pred)
    HD_T_pred = computeHD(_X_Ref, _TX_Pred)
    print("[RMSE] Root Mean Squared Surface Distance(mm): {:.4f}".format(RMSE_T_pred))
    print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(ASSD_T_pred))
    print("[HD] Hausdorff distance (mm): {:.4f}".format(HD_T_pred))

    Dice_VOE_lst = [utils.computeDiceAndVOE(_x_ref, _x_pred, pitch=0.2) for _x_ref, _x_pred in zip(_X_Ref, _TX_Pred)]
    avg_Dice, avg_VOE = np.array(Dice_VOE_lst).mean(axis=0)
    print("[DC] Volume Dice Coefficient: {:.4f}".format(avg_Dice))
    print("[VOE] Volumetric Overlap Error: {:.2f} %".format(100.*avg_VOE))

    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open(r'./dataWithPhoto/temp_result.csv', 'a', newline='') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow([patientID,RMSE_pred,ASSD_pred,HD_pred,RMSE_T_pred,ASSD_T_pred,HD_T_pred,avg_Dice,avg_VOE])
        f_object.close()


@ray.remote
def createAlignedPredMeshes(TagID):
    patientID = TagID
    demoH5File2Load = r"./dataWithPhoto/demo/demo_TagID={}.h5".format(patientID)
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

    TX_Pred_Upper_Meshes = [utils.surfaceVertices2WatertightO3dMesh(pg) for pg in TX_Pred_Upper]
    TX_Pred_Lower_Meshes = [utils.surfaceVertices2WatertightO3dMesh(pg) for pg in TX_Pred_Lower]
    Aligned_Pred_Upper_Mesh = utils.mergeO3dTriangleMeshes(TX_Pred_Upper_Meshes)
    Aligned_Pred_Lower_Mesh = utils.mergeO3dTriangleMeshes(TX_Pred_Lower_Meshes)

    demoMeshDir = r"./dataWithPhoto/demoMesh/{}/".format(patientID)
    if not os.path.exists(demoMeshDir):
        os.makedirs(demoMeshDir)
    utils.exportTriMeshObj(np.asarray(Ref_Upper_Mesh.vertices), np.asarray(Ref_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Ref_Upper_Mesh_TagID={}.obj".format(patientID)))
    utils.exportTriMeshObj(np.asarray(Ref_Lower_Mesh.vertices), np.asarray(Ref_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Ref_Lower_Mesh_TagID={}.obj".format(patientID)))
    utils.exportTriMeshObj(np.asarray(Aligned_Pred_Upper_Mesh.vertices), np.asarray(Aligned_Pred_Upper_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Aligned_Pred_Upper_Mesh_TagID={}.obj".format(patientID)))
    utils.exportTriMeshObj(np.asarray(Aligned_Pred_Lower_Mesh.vertices), np.asarray(Aligned_Pred_Lower_Mesh.triangles), \
                        os.path.join(demoMeshDir,"Aligned_Pred_Lower_Mesh_TagID={}.obj".format(patientID)))
    print("Finish create mesh of TagID: {}".format(patientID))


if __name__ == "__main__":
    # for TagID in TagIDs:
    #     LogFile = os.path.join(LOG_DIR, "TagID-{}.log".format(TagID))
    #     # Log file
    #     if os.path.exists(LogFile):
    #         os.remove(LogFile)
    #     log = open(LogFile, "a", encoding='utf-8')
    #     sys.stdout = log

    #     run_emopt(TagID)
    #     evaluation(TagID)

    #     log.close()

    # create demo triangle meshes
    NUM_CPUS = psutil.cpu_count(logical=False) 
    ray.init(num_cpus=NUM_CPUS, num_gpus=1) #ray(多线程)初始化
    ray.get([createAlignedPredMeshes.remote(TagID) for TagID in TagIDs])