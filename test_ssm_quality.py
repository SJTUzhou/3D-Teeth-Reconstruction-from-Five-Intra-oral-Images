import open3d as o3d
import numpy as np
import shutil
import os
import h5py
import glob
import pandas as pd
import matplotlib.pyplot as plt
import ssm_utils
from recons_eval_metric import computeRMSE


TOOTH_INDICES = ssm_utils.UPPER_INDICES + ssm_utils.LOWER_INDICES

CPD_SSM_DIR = r"./data/cpdAlignedData/eigValVec/" 
CPD_ALIGNED_DIR = r"./data/cpdAlignedData/" 
CPD_PARAM_DIR = r"./data/params/" 

CPDGP_SSM_DIR = r"./data/cpdGpAlignedData/eigValVec/" 
CPDGP_ALIGNED_DIR = r"./data/cpdGpAlignedData/" 
CPDGP_PARAM_DIR = r"./data/cpdGpParams/" 


def get_idx_by_tooth_type():
    i_incisor = []
    i_canine = []
    i_premolar = []
    i_molar = []
    for i,tid in enumerate(TOOTH_INDICES):
        t = i % 10
        if t == 1 or t == 2:
            i_incisor.append(i)
        elif t == 3:
            i_canine.append(i)
        elif t == 4 or t == 5:
            i_premolar.append(i)
        else:
            i_molar.append(i)
    return i_incisor, i_canine, i_premolar, i_molar


def Specificity(loadDir, num_sample=100):
    alignedPointGroups, txtFileIndices = ssm_utils.loadAlignedPointGroupsWithIndex(loadDir)
    eigVal, eigVec, normalizedTrainPointVector, meanTrainPointVector = ssm_utils.getEigValVecOfSSMByPCA(alignedPointGroups)
    explained_variance = np.cumsum(eigVal)/np.sum(eigVal)
    num_mode = len(eigVal)
    # num_mode = np.argmin(explained_variance<=0.95) + 1
    
    rnd_feature_vecs = np.random.normal(size=(num_sample, num_mode))
    eigVal = eigVal[:num_mode]
    eigVec = eigVec[:,:num_mode]
    
    rnd_samples = meanTrainPointVector + np.matmul(rnd_feature_vecs * eigVal, eigVec.T)
    rnd_samples = rnd_samples.reshape(num_sample, -1, 3)
    
    RMSE_rets = []
    for j in range(num_sample):
        rmses = [computeRMSE(rnd_samples[j], x) for x in alignedPointGroups]
        RMSE_rets.append(np.min(rmses))
    return np.array(RMSE_rets)



def specificity_all_teeth(srcDir, num_sample=100):
    num_teeth = len(TOOTH_INDICES)
    ret = np.zeros((num_teeth,num_sample))
    for i,tid in enumerate(TOOTH_INDICES):
        loadDir = os.path.join(srcDir, str(tid))
        ret[i,:] = Specificity(loadDir, num_sample)
    df = pd.DataFrame(ret.T, columns=[str(tid) for tid in TOOTH_INDICES])
    # print(df.describe())
    print("mean RMSE: {:.4f} mm".format(np.mean(ret)))
    print("std RMSE: {:.4f} mm".format(np.std(ret)))
    # plt.hist(ret.flatten(), bins=50)
    # plt.show()
    i_incisor, i_canine, i_premolar, i_molar = get_idx_by_tooth_type()
    print(f"Incisor - mean RMSE: {np.mean(ret[i_incisor]):.4f} mm - std RMSE: {np.std(ret[i_incisor]):.4f} mm")
    print(f"Canine - mean RMSE: {np.mean(ret[i_canine]):.4f} mm - std RMSE: {np.std(ret[i_canine]):.4f} mm")
    print(f"Premolar - mean RMSE: {np.mean(ret[i_premolar]):.4f} mm - std RMSE: {np.std(ret[i_premolar]):.4f} mm")
    print(f"molar - mean RMSE: {np.mean(ret[i_molar]):.4f} mm - std RMSE: {np.std(ret[i_molar]):.4f} mm")



def Generalization(loadDir):
    alignedPointGroups, txtFileIndices = ssm_utils.loadAlignedPointGroupsWithIndex(loadDir)
    num_sample = len(alignedPointGroups)
    RMSE_rets = []
    for i in range(num_sample):
        x = alignedPointGroups[i]
        leave_one_out_pgs = alignedPointGroups[:i] + alignedPointGroups[i+1:]
        eigVal, eigVec, normalizedTrainPointVector, meanTrainPointVector = ssm_utils.getEigValVecOfSSMByPCA(leave_one_out_pgs)
        explained_variance = np.cumsum(eigVal)/np.sum(eigVal)
        num_mode = np.argmin(explained_variance<=0.95) + 1
        eigVal = eigVal[:num_mode]
        eigVec = eigVec[:,:num_mode]
        
        normalized_x = x.flatten() - meanTrainPointVector
        featureVecs = normalized_x @ eigVec
        recons_x = meanTrainPointVector + featureVecs @ eigVec.T
        recons_x = recons_x.reshape(-1,3)
        RMSE_rets.append(computeRMSE(x, recons_x))
    return np.array(RMSE_rets)


def generalization_all_teeth(srcDir):
    rets = []
    for tid in TOOTH_INDICES:
        loadDir = os.path.join(srcDir, str(tid))
        ret = Generalization(loadDir)
        rets.append(ret)
        
    ret_data = np.concatenate(rets)
    print("mean RMSE: {:.4f} mm".format(np.mean(ret_data)))
    print("std RMSE: {:.4f} mm".format(np.std(ret_data)))
    
    i_incisor, i_canine, i_premolar, i_molar = get_idx_by_tooth_type()
    _rets = np.array(rets)
    print(f"Incisor - mean RMSE: {np.mean(np.concatenate(_rets[i_incisor])):.4f} mm - std RMSE: {np.std(np.concatenate(_rets[i_incisor])):.4f} mm")
    print(f"Canine - mean RMSE: {np.mean(np.concatenate(_rets[i_canine])):.4f} mm - std RMSE: {np.std(np.concatenate(_rets[i_canine])):.4f} mm")
    print(f"Premolar - mean RMSE: {np.mean(np.concatenate(_rets[i_premolar])):.4f} mm - std RMSE: {np.std(np.concatenate(_rets[i_premolar])):.4f} mm")
    print(f"molar - mean RMSE: {np.mean(np.concatenate(_rets[i_molar])):.4f} mm - std RMSE: {np.std(np.concatenate(_rets[i_molar])):.4f} mm")




def Compactness(loadDir, plot=False):
    alignedPointGroups, txtFileIndices = ssm_utils.loadAlignedPointGroupsWithIndex(loadDir)
    eigVal, eigVec, normalizedTrainPointVector, meanTrainPointVector = ssm_utils.getEigValVecOfSSMByPCA(alignedPointGroups)
    if plot == True:
        ssm_utils.visualizeCompactnessOfSSM(eigVal)
    explained_variance = np.cumsum(eigVal)/np.sum(eigVal)
    return explained_variance

def compactness_all_teeth(srcDir, plot=True):
    num_max = 0
    num_teeth = len(TOOTH_INDICES)
    explained_variances = []
    for tid in TOOTH_INDICES:
        loadDir = os.path.join(srcDir, str(tid))
        exp_var = Compactness(loadDir, plot=False)
        num_max = max(num_max, len(exp_var))
        explained_variances.append(exp_var)
    data_plot = np.ones((num_teeth, num_max+1), np.float64)
    data_plot[:,0] = 0.
    for i,exp_var in enumerate(explained_variances):
        k = len(exp_var)
        data_plot[i,1:k+1] = exp_var
    x = np.arange(0,num_max+1)
    ymin = np.min(data_plot, 0)
    ymax = np.max(data_plot, 0)
    ymean = np.mean(data_plot, 0)

    if plot == True:
        plt.figure(figsize=(8, 6))
        plt.ylim([0.,1.])
        plt.xlim([0.,num_max])
        plt.fill_between(x,ymin,ymax,color="orange")
        plt.plot(x, ymean, linewidth=2,color='darkorange')
        plt.ylabel("Cumulative explained variance")
        plt.xlabel("Num of variation Mode")
        plt.title("Compactness of PDM")
        plt.grid()
        plt.show()
    return data_plot


def compactnes_comparison(srcDir1, srcDir2, label1, label2):
    data_plot1 = compactness_all_teeth(srcDir1, plot=False)
    data_plot2 = compactness_all_teeth(srcDir2, plot=False)
    num_ = data_plot1.shape[1]
    x = np.arange(0,num_)
    ymin1 = np.min(data_plot1, 0)
    ymax1 = np.max(data_plot1, 0)
    ymean1 = np.mean(data_plot1, 0)
    ymin2 = np.min(data_plot2, 0)
    ymax2 = np.max(data_plot2, 0)
    ymean2 = np.mean(data_plot2, 0)
    
    plt.figure(figsize=(8, 6))
    plt.ylim([0.,1.])
    plt.xlim([0.,num_])
    plt.fill_between(x,ymin1,ymax1,color="orange",alpha=0.5,label=label1)
    plt.plot(x, ymean1, linewidth=2,color='darkorange')
    plt.fill_between(x,ymin2,ymax2,color="deepskyblue",alpha=0.5,label=label2)
    plt.plot(x, ymean2, linewidth=2,color='dodgerblue')
    plt.ylabel("Cumulative Explained Variance")
    plt.xlabel("Num of Eigenmodes")
    plt.title("Compactness of PDM")
    plt.grid()
    plt.legend()
    # plt.show()
    
    fig = plt.gcf()
    fig.set_size_inches(8,4)
    plt.savefig('compactness.png', dpi=300)
    
    

if __name__ == "__main__":
    compactnes_comparison(srcDir1=CPD_ALIGNED_DIR, srcDir2=CPDGP_ALIGNED_DIR, \
        label1="With point correspondence established by CPD-based algo.", label2="With point correspondence established by CPD-GP-based algo.")
    
    # specificity_all_teeth(CPD_ALIGNED_DIR, num_sample=1000)
    # specificity_all_teeth(CPDGP_ALIGNED_DIR, num_sample=1000)
    # mean RMSE: 1.3986 mm
    # std RMSE: 0.8339 mm
    # Incisor - mean RMSE: 1.2176 mm - std RMSE: 0.6278 mm
    # Canine - mean RMSE: 1.2989 mm - std RMSE: 0.6842 mm
    # Premolar - mean RMSE: 1.2342 mm - std RMSE: 0.7186 mm
    # molar - mean RMSE: 1.5810 mm - std RMSE: 0.9519 mm
    # mean RMSE: 1.8100 mm
    # std RMSE: 1.1741 mm
    # Incisor - mean RMSE: 1.4165 mm - std RMSE: 0.7568 mm
    # Canine - mean RMSE: 2.1238 mm - std RMSE: 1.2382 mm
    # Premolar - mean RMSE: 1.4103 mm - std RMSE: 0.8160 mm
    # molar - mean RMSE: 2.1037 mm - std RMSE: 1.3377 mm


    
    
    # generalization_all_teeth(CPD_ALIGNED_DIR)
    # generalization_all_teeth(CPDGP_ALIGNED_DIR)
    # mean RMSE: 0.1912 mm
    # std RMSE: 0.0481 mm
    # Incisor - mean RMSE: 0.1801 mm - std RMSE: 0.0405 mm
    # Canine - mean RMSE: 0.1891 mm - std RMSE: 0.0405 mm
    # Premolar - mean RMSE: 0.1877 mm - std RMSE: 0.0447 mm
    # molar - mean RMSE: 0.1985 mm - std RMSE: 0.0530 mm
    # mean RMSE: 0.1337 mm
    # std RMSE: 0.0341 mm
    # Incisor - mean RMSE: 0.1230 mm - std RMSE: 0.0332 mm
    # Canine - mean RMSE: 0.1406 mm - std RMSE: 0.0337 mm
    # Premolar - mean RMSE: 0.1241 mm - std RMSE: 0.0240 mm
    # molar - mean RMSE: 0.1415 mm - std RMSE: 0.0361 mm