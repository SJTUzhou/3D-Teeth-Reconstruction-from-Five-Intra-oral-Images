import glob
import os
import numpy as np
import open3d as o3d
from const import *




def getToothIndex(f):
    return int(os.path.basename(f).split(".")[0].split("_")[-1])


def loadMuEigValSigma(ssmDir, numPC):
    """Mu.shape=(28,1500,3), sqrtEigVals.shape=(28,1,100), Sigma.shape=(28,4500,100)"""
    muNpys = glob.glob(os.path.join(ssmDir,"meanAlignedPG_*.npy"))
    muNpys = sorted(muNpys, key=lambda x:getToothIndex(x))
    Mu = np.array([np.load(x) for x in muNpys])
    eigValNpys = glob.glob(os.path.join(ssmDir,"eigVal_*.npy"))
    eigValNpys = sorted(eigValNpys, key=lambda x:getToothIndex(x))
    sqrtEigVals = np.sqrt(np.array([np.load(x) for x in eigValNpys]))
    eigVecNpys = glob.glob(os.path.join(ssmDir,"eigVec_*.npy"))
    eigVecNpys = sorted(eigVecNpys, key=lambda x:getToothIndex(x))
    Sigma = np.array([np.load(x) for x in eigVecNpys])
    return Mu, sqrtEigVals[:,np.newaxis,:numPC], Sigma[...,:numPC]





def read_demo_mesh_vertices_by_FDI(mesh_dir, tag, FDIs):
    mesh_vertices_by_FDI = []
    for fdi in FDIs:
        mshf = os.path.join(mesh_dir, str(tag), "byFDI", f"Ref_Mesh_Tag={tag}_FDI={fdi}.obj")
        msh = o3d.io.read_triangle_mesh(mshf)
        mesh_vertices_by_FDI.append(np.asarray(msh.vertices, np.float64))
    return mesh_vertices_by_FDI


        
    


