import os
import numpy as np
import pandas as pd
import pcd_mesh_utils
import recons_eval_metric as metric
from ssm_utils import UPPER_INDICES, LOWER_INDICES
import projection_utils as proj
import ray
import psutil


SRC_DIR = r"./data/cpdGpAlignedData/"
ARRAY_SAVE_DIR = r"./data/cpdGpParams/"
REF_DIR = r"./dataWithPhoto/cpdGpParams/"
PG_NPY = os.path.join(REF_DIR, "Y_pg.npy")
MASK_NPY = os.path.join(REF_DIR, "X_mask.npy")
PG_SHAPE = (1500,3)
NUM_SSM_SAMPLE = 130
NEAREST_RETRIEVAL_CSV = r"./dataWithPhoto/_temp/evaluation_nearest_retrieval.csv"
TOOTH_INDICES = UPPER_INDICES + LOWER_INDICES

def loadSSMDataSet(pgShape=PG_SHAPE, srcDir=SRC_DIR):
    pgsU = []
    pgsL = []
    masksU = []
    masksL = []
    for i in range(NUM_SSM_SAMPLE): # indices used for SSM
        maskU, alignedPGU = proj.loadAlignedToothRowPgWithMask(i, pgShape, srcDir, "U")
        maskL, alignedPGL = proj.loadAlignedToothRowPgWithMask(i, pgShape, srcDir, "L")
        pgsU.append(alignedPGU)
        masksU.append(maskU)
        pgsL.append(alignedPGL)
        masksL.append(maskL)
    return np.array(pgsU), np.array(pgsL), np.array(masksU), np.array(masksL)



@ray.remote
class SsmPgActor:
    def __init__(self, pgsU, pgsL, masksU, masksL) -> None:
        self._pgsU = pgsU
        self._pgsL = pgsL
        self._masksU = masksU
        self._masksL = masksL
        self.numItem = len(self._masksU)
    
    def getPgMask(self, i):
        assert i>=0 and i<self.numItem
        # print(self._pgsU)
        return self._pgsU[i], self._pgsL[i], self._masksU[i], self._masksL[i]

@ray.remote
def nearest_retrieval(tagID, ssmPgActor):
    Mask = proj.GetMaskByTagId(MASK_NPY, tagID)
    indices = np.array(TOOTH_INDICES)[Mask]
    PG_Ref = proj.GetPGByTagId(PG_NPY, tagID)
    Mask_U, Mask_L = np.split(Mask, 2, axis=0)
    X_Ref_U, X_Ref_L = proj.GetPgRefUL(PG_Ref, Mask)
    X_Ref = PG_Ref[Mask]
    RMSEs = []
    for i in range(NUM_SSM_SAMPLE):
        pgU, pgL, maskU, maskL = ray.get(ssmPgActor.getPgMask.remote(i))
        _mask = np.hstack([maskU, maskL])
        if not np.all(_mask[Mask]):
            RMSEs.append(np.inf)
            continue # exists missing teeth in i-th SSM sample compared with ref Pg
        _pgU = pgU[Mask_U]
        _pgL = pgL[Mask_L]
        T_Upper = pcd_mesh_utils.computeTransMatByCorres(_pgU.reshape(-1,3), X_Ref_U.reshape(-1,3), with_scale=False)
        T_Lower = pcd_mesh_utils.computeTransMatByCorres(_pgL.reshape(-1,3), X_Ref_L.reshape(-1,3), with_scale=False)
        T_pgU = np.matmul(_pgU, T_Upper[:3,:3]) + T_Upper[3,:3]
        T_pgL = np.matmul(_pgL, T_Lower[:3,:3]) + T_Lower[3,:3]
        rmse = metric.computeRMSE(X_Ref, np.concatenate([T_pgU,T_pgL]))
        RMSEs.append(rmse)
    
    j = np.argmin(RMSEs)
    pgU, pgL, maskU, maskL = ray.get(ssmPgActor.getPgMask.remote(j))
    _pgU = pgU[Mask_U]
    _pgL = pgL[Mask_L]
    T_Upper = pcd_mesh_utils.computeTransMatByCorres(_pgU.reshape(-1,3), X_Ref_U.reshape(-1,3), with_scale=True)
    T_Lower = pcd_mesh_utils.computeTransMatByCorres(_pgL.reshape(-1,3), X_Ref_L.reshape(-1,3), with_scale=True)
    T_pgU = np.matmul(_pgU, T_Upper[:3,:3]) + T_Upper[3,:3]
    T_pgL = np.matmul(_pgL, T_Lower[:3,:3]) + T_Lower[3,:3]
    T_pg = np.concatenate([T_pgU,T_pgL])

    RMSDs = np.array(metric.computeRMSD(X_Ref, T_pg, return_list=True))
    ASSDs = np.array(metric.computeASSD(X_Ref, T_pg, return_list=True))
    HDs = np.array(metric.computeHD(X_Ref, T_pg, return_list=True))
    Dice_VOE_lst = [metric.computeDiceAndVOE(_x1, _x2, pitch=0.2) for _x1, _x2 in zip(X_Ref, T_pg)]
    Dice_VOEs = np.array(Dice_VOE_lst)

    Metrics_array = np.hstack([np.tile(tagID,(len(indices),1)), indices[:,None], RMSDs[:,None], ASSDs[:,None], HDs[:,None], Dice_VOEs])
    df = pd.DataFrame(data=Metrics_array, columns=["tagID", "toothID", "RMSD", "ASSD", "HD", "DSC", "VOE"])
    df = df.astype({"tagID":int,"toothID":int})
    print("Finish TagID: ", tagID)
    return df





if __name__ == "__main__":
    arraySaveDir = ARRAY_SAVE_DIR
    pgU_npy = os.path.join(arraySaveDir, "SSM_PG_U.npy")
    pgL_npy = os.path.join(arraySaveDir, "SSM_PG_L.npy")
    maskU_npy = os.path.join(arraySaveDir, "SSM_MASK_U.npy")
    maskL_npy = os.path.join(arraySaveDir, "SSM_MASK_L.npy")

    # pgsU, pgsL, masksU, masksL = loadSSMDataSet(pgShape=PG_SHAPE, srcDir=SRC_DIR)
    # np.save(pgU_npy, pgsU)
    # np.save(pgL_npy, pgsL)
    # np.save(maskU_npy, masksU)
    # np.save(maskL_npy, masksL)

    pgsU = np.load(pgU_npy)
    pgsL = np.load(pgL_npy)
    masksU = np.load(maskU_npy)
    masksL = np.load(maskL_npy)

    NUM_CPUS = psutil.cpu_count(logical=False)
    ray.init(num_cpus=NUM_CPUS, num_gpus=1) #ray(多线程)初始化

    ssmPgActor = SsmPgActor.remote(pgsU, pgsL, masksU, masksL)

    metric_DFs = ray.get([nearest_retrieval.remote(tagID, ssmPgActor) for tagID in range(0,95)])
    df = pd.concat(metric_DFs, ignore_index=True)
    df.to_csv(NEAREST_RETRIEVAL_CSV, mode='a', index=False, header=True)


