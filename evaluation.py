import os
import sys
from tracemalloc import Statistic
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import scipy.io
import utils
import projection_utils as proj
import ray
import psutil
from utils import UPPER_INDICES, LOWER_INDICES


SSM_DIR = r"./data/cpdGpAlignedData/eigValVec/" 
NUM_PC = 10
DEMO_H5_DIR = r"./dataWithPhoto/demo/Grad-99%conf-v21-PC=10/"
EVAL_SSM_CSV = r"./dataWithPhoto/_temp/evaluation_SSM.csv"
EVAL_3D_RE_CSV = r"./dataWithPhoto/_temp/evaluation_3Dre.csv"
EVAL_3D_RE_STATIS_CSV = r"./dataWithPhoto/_temp/3Dre_statistics.csv"
TOOTH_INDICES = UPPER_INDICES + LOWER_INDICES

@ray.remote
def evaluate_SSM(TagID, Mu, Sigma):
    demoH5File2Load = os.path.join(DEMO_H5_DIR, "demo_TagID={}.h5".format(TagID))
    with h5py.File(demoH5File2Load, 'r') as f:
        grp = f[str(TagID)]
        X_Ref_Upper = grp["UPPER_REF"][:]
        X_Ref_Lower = grp["LOWER_REF"][:]
        Mask = np.array(grp["MASK"][:], dtype=np.bool_)
    _X_Ref = np.concatenate([X_Ref_Upper,X_Ref_Lower])
    indices = np.array(TOOTH_INDICES)[Mask]
    _Mu = Mu[Mask]
    _Sigma = Sigma[Mask]
    _Re_X_Ref = []
    for x_ref, mu, eigVecs in zip(_X_Ref,_Mu,_Sigma):
        _T = utils.computeTransMatByCorres(x_ref, mu, True)
        _sR = _T[:3,:3] # 右乘矩阵
        _t = _T[3,:3]
        Tx_ref = np.matmul(x_ref, _sR) + _t #(1500,3)
        _deform_vec = Tx_ref.flatten() - mu.flatten() #(4500,)
        _feature_vecs = np.matmul(_deform_vec[None,:], eigVecs) #(1,10)
        re_Tx_ref = np.matmul(_feature_vecs, eigVecs.T).reshape(-1,3) + mu #(1500,3)
        re_x_ref = np.matmul((re_Tx_ref - _t), np.linalg.inv(_sR)) #(1500,3)
        _Re_X_Ref.append(re_x_ref)
    _Re_X_Ref = np.array(_Re_X_Ref)
    RMSDs = np.array(utils.computeRMSD(_X_Ref, _Re_X_Ref, return_list=True))
    ASSDs = np.array(utils.computeASSD(_X_Ref, _Re_X_Ref, return_list=True))
    HDs = np.array(utils.computeHD(_X_Ref, _Re_X_Ref, return_list=True))
    Dice_VOE_lst = [utils.computeDiceAndVOE(_x1, _x2, pitch=0.2) for _x1, _x2 in zip(_X_Ref, _Re_X_Ref)]
    Dice_VOEs = np.array(Dice_VOE_lst)
    Metrics_array = np.hstack([np.tile(TagID,(len(indices),1)), indices[:,None], RMSDs[:,None], ASSDs[:,None], HDs[:,None], Dice_VOEs])
    df = pd.DataFrame(data=Metrics_array, columns=["tagID", "toothID", "RMSD", "ASSD", "HD", "DSC", "VOE"])
    df = df.astype({"tagID":int,"toothID":int})
    print("Finish TagID: ", TagID)
    return df


@ray.remote
def evaluation_3D_reconstruction(TagID):
    demoH5File2Load = os.path.join(DEMO_H5_DIR, "demo_TagID={}.h5".format(TagID))
    with h5py.File(demoH5File2Load, 'r') as f:
        grp = f[str(TagID)]
        X_Pred_Upper = grp["UPPER_PRED"][:]
        X_Pred_Lower = grp["LOWER_PRED"][:]
        X_Ref_Upper = grp["UPPER_REF"][:]
        X_Ref_Lower = grp["LOWER_REF"][:]
        Mask = np.array(grp["MASK"][:], dtype=np.bool_)
    _X_Ref = np.concatenate([X_Ref_Upper,X_Ref_Lower])
    indices = np.array(TOOTH_INDICES)[Mask]

    # 相似变换配准后的牙列预测与Ground Truth对比
    with_scale = True
    T_Upper = utils.computeTransMatByCorres(X_Pred_Upper.reshape(-1,3), X_Ref_Upper.reshape(-1,3), with_scale=with_scale)
    T_Lower = utils.computeTransMatByCorres(X_Pred_Lower.reshape(-1,3), X_Ref_Lower.reshape(-1,3), with_scale=with_scale)

    TX_Pred_Upper = np.matmul(X_Pred_Upper, T_Upper[:3,:3]) + T_Upper[3,:3]
    TX_Pred_Lower = np.matmul(X_Pred_Lower, T_Lower[:3,:3]) + T_Lower[3,:3]
    _TX_Pred = np.concatenate([TX_Pred_Upper, TX_Pred_Lower])

    RMSD_T_pred = np.array(utils.computeRMSD(_X_Ref, _TX_Pred, return_list=True))
    ASSD_T_pred = np.array(utils.computeASSD(_X_Ref, _TX_Pred, return_list=True))
    HD_T_pred = np.array(utils.computeHD(_X_Ref, _TX_Pred, return_list=True))
    Dice_VOE_lst = [utils.computeDiceAndVOE(_x_ref, _x_pred, pitch=0.2) for _x_ref, _x_pred in zip(_X_Ref, _TX_Pred)]
    Dice_VOE_T_pred = np.array(Dice_VOE_lst)

    Metrics_array = np.hstack([np.tile(TagID,(len(indices),1)), indices[:,None], RMSD_T_pred[:,None], ASSD_T_pred[:,None], HD_T_pred[:,None], Dice_VOE_T_pred])
    df = pd.DataFrame(data=Metrics_array, columns=["tagID", "toothID", "RMSD", "ASSD", "HD", "DSC", "VOE"])
    df = df.astype({"tagID":int,"toothID":int})
    print("Finish TagID: ", TagID)
    return df

def analyze_metrics_per_tooth_type(metric_csv):
    df = pd.read_csv(metric_csv)
    statistics_DFs = []
    for tID in TOOTH_INDICES:
        print("Tooth Index:", tID)
        _metrics = df.loc[df["toothID"]==tID]
        _statistics_df = _metrics[["RMSD", "ASSD", "HD", "DSC", "VOE"]].describe(percentiles=[.5])
        print(_statistics_df)
        _statistics_df["toothID"] = tID
        statistics_DFs.append(_statistics_df)
    df2 = pd.concat(statistics_DFs, axis=0)
    return df2

def analyze_metrics(metric_csv):
    df = pd.read_csv(metric_csv)
    # df = df.groupby("tagID").mean()
    statistics_df = df[["RMSD", "ASSD", "HD", "DSC", "VOE"]].describe(percentiles=[.5])
    print(statistics_df)
    return statistics_df.T

if __name__ == "__main__":
    # # 多线程并行计算三维重建各个Metric
    # NUM_CPUS = psutil.cpu_count(logical=False)
    # ray.init(num_cpus=NUM_CPUS, num_gpus=1) #ray(多线程)初始化
    # metric_DFs = ray.get([evaluation_3D_reconstruction.remote(tagID) for tagID in range(0,95)])
    # df = pd.concat(metric_DFs, ignore_index=True)
    # df.to_csv(EVAL_3D_RE_CSV, mode='a', index=False, header=True)

    # # 保存按牙齿分类的三维重建评估指标
    # df = analyze_metrics_per_tooth_type(EVAL_3D_RE_CSV)
    # df.to_csv(EVAL_3D_RE_STATIS_CSV, index=True, float_format='%.4f')

    # # 分析整体的三维重建评估指标
    # analyze_metrics(EVAL_3D_RE_CSV)

    # # 多线程SSM计算SSM评估指标
    # NUM_CPUS = psutil.cpu_count(logical=False)
    # ray.init(num_cpus=NUM_CPUS) #ray(多线程)初始化
    # Mu, sqrtEigVals, Sigma = proj.loadMuEigValSigma(SSM_DIR, numPC=NUM_PC) # Mu.shape=(28,1500,3), sqrtEigVals.shape=(28,1,10), Sigma.shape=(28,4500,10)
    # metric_DFs = ray.get([evaluate_SSM.remote(tagID,Mu,Sigma) for tagID in range(0,95)])
    # df = pd.concat(metric_DFs, ignore_index=True)
    # df.to_csv(EVAL_SSM_CSV, mode='a', index=False, header=True)

    analyze_metrics(EVAL_SSM_CSV)
