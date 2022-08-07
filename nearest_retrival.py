import os
import numpy as np
import pandas as pd
import utils
import projection_utils as proj


SRC_DIR = r"./data/cpdGpAlignedData/"
PARAM_DIR = r"./data/cpdGpParams/"
PG_SHAPE = (1500,3)

def loadSSMDataSet(pgShape=PG_SHAPE, srcDir=SRC_DIR):
    pgsU = []
    pgsL = []
    masksU = []
    masksL = []
    for i in range(0,130): # indices used for SSM
        maskU, alignedPGU = proj.loadAlignedToothRowPgWithMask(i, pgShape, srcDir, "U")
        maskL, alignedPGL = proj.loadAlignedToothRowPgWithMask(i, pgShape, srcDir, "L")
        pgsU.append(alignedPGU)
        masksU.append(maskU)
        pgsL.append(alignedPGL)
        masksL.append(maskL)
    return np.array(pgsU), np.array(pgsL), np.array(masksU), np.array(masksL)


if __name__ == "__main__":
    arraySaveDir = PARAM_DIR
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

    
