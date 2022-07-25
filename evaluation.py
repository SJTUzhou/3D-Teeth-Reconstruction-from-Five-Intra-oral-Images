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
import ray
import psutil
from utils import UPPER_INDICES, LOWER_INDICES



DEMO_H5_DIR = r"./dataWithPhoto/demo/Grad-99%conf-v21-PC=10/"
TEMP_CSV = r"./dataWithPhoto/_temp/evaluation_metrics.csv"
STATISTICS_CSV = r"./dataWithPhoto/_temp/metric_statistics.csv"
TOOTH_INDICES = UPPER_INDICES + LOWER_INDICES

@ray.remote
def evaluation(TagID):
    patientID = TagID
    demoH5File2Load = os.path.join(DEMO_H5_DIR, "demo_TagID={}.h5".format(patientID))
    with h5py.File(demoH5File2Load, 'r') as f:
        grp = f[str(patientID)]
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
    print(statistics_df.T)
    return statistics_df.T

if __name__ == "__main__":
    # # 多线程并行计算各个Metric
    # NUM_CPUS = psutil.cpu_count(logical=False)
    # ray.init(num_cpus=NUM_CPUS, num_gpus=1) #ray(多线程)初始化
    # metric_DFs = ray.get([evaluation.remote(tagID) for tagID in range(0,95)])
    # df = pd.concat(metric_DFs, ignore_index=True)
    # df.to_csv(TEMP_CSV, mode='a', index=False, header=True)

    # # 保存按牙齿分类的统计量
    # df = analyze_metrics_per_tooth_type(TEMP_CSV)
    # df.to_csv(STATISTICS_CSV, index=True, float_format='%.4f')

    analyze_metrics(TEMP_CSV)