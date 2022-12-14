import igl
import numpy as np
import trimesh
import os
import ray
import psutil
import open3d as o3d
import shutil
import time
from pcd_mesh_utils import voxelDownSample



"""修补1--7号牙齿三角面片的大面积缺损"""
"""边为平均边长3倍时分割，增加大三角网格上采样，边缘拉普拉斯平滑，增加整体拉普拉斯平滑，feasible_thre = 0.5"""
"""生成修补后的obj文件和牙齿点云下采样得到的txt文件"""

# 边界某点到点云中心的向量与这一点到其边界相邻2点连线的中点的向量的夹角的cos值是否小于一个值(经验值)FEASIBLE_COS_THRE
# FEASIBLE_COS_THRE = 0.6 # TOOTH 11---17,21---27
FEASIBLE_COS_THRE = 0.8 # TOOTH 31,32,41,42
EDGE_SEP_COEF = 3.0 #新生成的边长大于平均边长此倍数会被二分

def getMesh(objMeshPath):
    """读取obj文件获取牙齿三角面片网格，判断是否缺损，缺损边界点的index，所有的顶点和三角面"""
    v, _, n, f, _, _ = igl.read_obj(objMeshPath, dtype='float32')
    if n.size > 0:
        print("{} has defined normal vectors.".format(objMeshPath))
    return v, f


def update_v_bd(v_bd, v_bd_id, index2remove, insertFlag, new_vs, new_v_ids):
    """在修补过程中，更新当前缺损边界的点的位置和对应的index"""
    if insertFlag: # 有新顶点加入
        v_bd = np.vstack([v_bd[:index2remove],new_vs,v_bd[index2remove+1:]])
        v_bd_id = np.hstack([v_bd_id[:index2remove],new_v_ids,v_bd_id[index2remove+1:]])
    else:
        v_bd = np.vstack([v_bd[:index2remove],v_bd[index2remove+1:]])
        v_bd_id = np.hstack([v_bd_id[:index2remove],v_bd_id[index2remove+1:]])
    return v_bd, v_bd_id


def get_bd_ln_rn(v_bd):
    """获取边界点的相邻两组点，ln:left neighbour，实际方向未知"""
    v_bd_ln = np.vstack([v_bd[-1], v_bd[:-1]]) # left neighbor of current index
    v_bd_rn = np.vstack([v_bd[1:], v_bd[0]]) # right neighbor of current index
    return v_bd_ln, v_bd_rn


def get_min_angle_index(v_bd, v_bd_ln, v_bd_rn, v_center):
    """找到夹角最小的点在v_bd中的index，和边界上边的平均长度"""
    bd_lv = v_bd_ln - v_bd # boundary left vector
    bd_rv = v_bd_rn - v_bd
    bd_lv_norms = np.linalg.norm(bd_lv, ord=2, axis=1) # default L2 norm for vector
    bd_rv_norms = np.hstack([bd_lv_norms[1:],bd_lv_norms[0]]) # 等价于 bd_rv_norms = np.linalg.norm(bd_rv, axis=1)
    cos_v_bd = np.sum(bd_lv*bd_rv, axis=1) / (bd_lv_norms * bd_rv_norms)
    
    # 排除反向边界点（与相邻边夹角大于180度）
    bd_cv = bd_lv + bd_rv # bd_cv可能是零向量
    bd_cv_norms = np.linalg.norm(bd_cv, ord=2, axis=1) #边界上一点到其相邻两点中点的距离
    valid_v_bd_mask = bd_cv_norms > 1e-2
    valid_v_bd = v_bd[valid_v_bd_mask] # 所有不与左右邻点共线的顶点
    valid_bd_2_centerV = v_center-valid_v_bd
    valid_bd_2_centerV_norms = np.linalg.norm(valid_bd_2_centerV, ord=2, axis=1) #边界上一[有效]点到点云中点的距离
    valid_cos_bd_cv_bd2centerV = np.sum(bd_cv[valid_v_bd_mask]*valid_bd_2_centerV, axis=1) / (bd_cv_norms[valid_v_bd_mask]*valid_bd_2_centerV_norms)
    # 边界某点到点云中心的向量与这一点到其边界相邻2点连线的中点的向量的夹角的cos值是否小于一个值(经验值)

    feasible_thre = FEASIBLE_COS_THRE # 这是一个经验值 cos值小于它被视为feasible
    feasible_valid_v_bd_mask = valid_cos_bd_cv_bd2centerV < feasible_thre 
    while not feasible_valid_v_bd_mask.any(): # 所有不与左右邻点共线的顶点的上述夹角都不满足条件
        feasible_thre += 0.02
        feasible_valid_v_bd_mask = valid_cos_bd_cv_bd2centerV < feasible_thre
   
    cos_v_bd[valid_v_bd_mask] = cos_v_bd[valid_v_bd_mask] + 2*feasible_valid_v_bd_mask
    min_angle_index = np.argmax(cos_v_bd)
    return min_angle_index


def getCoarsePatch(v_bd_id_init, v, f):
    """ 生成初步的粗糙的mesh补丁
    实现算法依据 https://www.cnblogs.com/shushen/p/5759679.html"""
    v_bd_id = v_bd_id_init.copy()
    v_center = np.mean(v, axis=0)
    v_bd = v[v_bd_id]
    new_v_lst = []
    new_v_id = np.max(f) + 1
    new_face_lst = []
    avg_edge_l = igl.avg_edge_length(v, f)
    while v_bd_id.size > 3:
        # print("number of boundary points: ", v_bd_id.size)
        v_bd_ln, v_bd_rn = get_bd_ln_rn(v_bd)
        min_angle_index = get_min_angle_index(v_bd, v_bd_ln, v_bd_rn, v_center)
        ln_rn_edge_l = np.linalg.norm(v_bd_ln[min_angle_index]-v_bd_rn[min_angle_index], ord=2) #这个点左右相邻点的距离
        v_bd_id_c, v_bd_id_l, v_bd_id_r = v_bd_id[min_angle_index], v_bd_id[min_angle_index-1], v_bd_id[(min_angle_index+1)%v_bd_id.size]
        edgeRatio = np.floor(ln_rn_edge_l/(EDGE_SEP_COEF*avg_edge_l)) # 3 测试
        if edgeRatio <= 1.0: # v_bd[min_angle_index] #边界上与左右顶点夹角最小的点
            new_face = (v_bd_id[min_angle_index], v_bd_id[min_angle_index-1], v_bd_id[(min_angle_index+1)%v_bd_id.size]) # 注意三角面片中的点的方向保持一致
            new_face_lst.append(new_face)
            v_bd, v_bd_id = update_v_bd(v_bd, v_bd_id, index2remove=min_angle_index, insertFlag=False, new_vs=None, new_v_ids=None)
        else:

            # new_v = (v_bd_rn[min_angle_index] + v_bd_ln[min_angle_index]) / 2.0
            # temp_new_vs = [new_v, ]
            # temp_new_v_ids = [new_v_id, ]
            # new_v_lst.append(new_v)
            # new_face_1 = (v_bd_id_c, v_bd_id_l, new_v_id)
            # new_face_2 = (v_bd_id_c, new_v_id, v_bd_id_r)
            # new_face_lst.append(new_face_1)
            # new_face_lst.append(new_face_2)
            # new_v_id += 1

            step_diff_v_l2r = (v_bd_rn[min_angle_index] - v_bd_ln[min_angle_index])/edgeRatio
            temp_new_vs = []
            temp_new_v_ids = []
            for i in range(1, int(edgeRatio)):
                new_v = v_bd_ln[min_angle_index] + i*step_diff_v_l2r
                # new_v = new_v + 0.2*(new_v - v_bd[min_angle_index]) # 测试
                temp_new_vs.append(new_v)
                temp_new_v_ids.append(new_v_id)
                if i==1:
                    new_face = (v_bd_id_c, v_bd_id_l, new_v_id) # 注意三角面片中的点的方向保持一致
                    new_face_lst.append(new_face)
                else:
                    new_face = (v_bd_id_c, new_v_id-1, new_v_id) # 注意三角面片中的点的方向保持一致
                    new_face_lst.append(new_face)
                if i==edgeRatio-1: # last face in this expansion
                    new_face = (v_bd_id_c, new_v_id, v_bd_id_r) # 注意三角面片中的点的方向保持一致
                    new_face_lst.append(new_face)
                new_v_id += 1
            new_v_lst.extend(temp_new_vs)
            v_bd, v_bd_id = update_v_bd(v_bd, v_bd_id, index2remove=min_angle_index,\
                 insertFlag=True, new_vs=np.array(temp_new_vs), new_v_ids=np.array(temp_new_v_ids))
            
    # print("number of boundary points: ", v_bd_id.size)
    if v_bd_id.size == 3:
        new_face_lst.append(tuple(v_bd_id))
    return np.array(new_v_lst, dtype=np.float32), np.array(new_face_lst, dtype=np.int32), np.arange(np.max(f)+1, new_v_id, dtype=np.int32)


def refineMeshByFirstOrderLaplacian(F, v_modif_id, v_control_id, v):
    """使用Least Square Mesh方法优化顶点位置，等价于具有控制顶点的1阶Laplacian优化
    F: 三角面片网格，其中点的index需要包含v_modif_id和v_control_id
    v_modif_id: 1阶Laplacian优化中待优化的点的index
    v_control_id: 1阶Laplacian优化中控制点的index
    v: 顶点坐标array，对应index处的坐标需与v_modif_id, v_control_id相对应
    Return: v_modif_id对应顶点更新后的坐标"""
    control_v = v[v_control_id]
    __adjMat = igl.adjacency_matrix(F)
    corr_v_id = np.hstack([v_modif_id, v_control_id])
    adjMat = __adjMat[corr_v_id,:][:, corr_v_id]
    neighCount = np.sum(adjMat, axis=1)
    __diagMat = np.diagflat(neighCount)
    laplacianMat = (__diagMat - adjMat) # 只包含v_modif_id, v_control_id的Graph Laplacian matrix
    numControlV = v_control_id.shape[0]
    numModifV = v_modif_id.shape[0]
    bdCondMat = np.hstack([np.zeros(shape=(numControlV, numModifV)), np.identity(numControlV)])
    matA = np.vstack([laplacianMat, bdCondMat])
    mat_b = np.vstack([np.zeros(shape=(numModifV+numControlV,3)),control_v])
    # vec_bx, vec_by, vec_bz = np.hsplit(mat_b, 3)
    # assert (vec_bx[numModifV:]==control_v[:,[0]]).all() #检查是否相等
    newModifV = np.linalg.inv(matA.T @ matA) @ matA.T @ mat_b #最小二乘法求解
    return newModifV[:numModifV]


def refineVertexPosFromNeighByFirstOrderLaplacian(f, v_modif_id, v):
    """用于对pathch与原始mesh的边界点进行1阶Laplacian优化"""
    adjMat = igl.adjacency_matrix(f)
    v_modif_adj_id = np.argwhere(np.array(adjMat[v_modif_id,:].sum(axis=0)).flatten()>0).flatten()
    v_control_id = np.array(list(set(v_modif_adj_id)-set(v_modif_id)))
    new_v_modif = refineMeshByFirstOrderLaplacian(f, v_modif_id, v_control_id, v)
    newV = v.copy()
    newV[v_modif_id] = new_v_modif
    return np.array(newV)
    

def repairFaceWindingOrderAndNormals(vertices, faces, outputObjFile):
    """如果初始修补的patch中的face winding order错误 需要重新修改"""
    # mesh2repair = trimesh.load_mesh(objFile2Repair) #从文件中读取
    mesh2repair = trimesh.Trimesh(vertices=vertices, faces=faces)
    # print("watertight mesh? ", mesh2repair.is_watertight) # 是否是水密三角网格
    # trimesh.repair.fill_holes(mesh2repair) # 修补上采樣后存在的孔洞
    # trimesh.repair.fix_winding(mesh2repair)
    trimesh.repair.fix_normals(mesh2repair, multibody=False) # 修改三角面片中顶点顺序，统一三角面片的winding方向 # Fix the winding and direction of a mesh face and face normals in-place
   
    exportStr = trimesh.exchange.obj.export_obj(mesh2repair, include_normals=False, include_color=False, include_texture=False, 
        return_texture=False, write_texture=False, resolver=None, digits=8)
    with open(outputObjFile,"w") as f:
        f.write(exportStr)
    return mesh2repair.vertices, mesh2repair.faces


@ray.remote
def generateOneRefinedPatchedTooth(srcDstFileTuple, keepPrev=False, voxel_size=0.2):
    """修补一颗牙齿，生成obj，txt文件"""
    srcObjFile, dstObjFile, dstTxtFile = srcDstFileTuple # 方便multiprocessing
    if os.path.exists(dstObjFile) and os.path.exists(dstTxtFile) and (keepPrev==True):
        print("Already Repair {}".format(srcObjFile))
        return 
    v, f = getMesh(srcObjFile)
    # print(v.shape)
    v_bd_id = igl.boundary_loop(f)
    if v_bd_id.size == 0:
        print("No hole exists in this mesh.")
        return
    
    avg_edge_length = igl.avg_edge_length(v, f)
    # v = igl.per_vertex_attribute_smoothing(v, f) # 整体 uniform laplacian smoothing
    print("Start repairing {}".format(srcObjFile))
    while v_bd_id.size > 0:
        # print("------")
        patchV, patchF, patchV_id = getCoarsePatch(v_bd_id, v, f)
        if patchF.size==0:
            break
        f = np.vstack([f,patchF])
        if patchV.size > 0:
            v = np.vstack([v,patchV])
            v = refineVertexPosFromNeighByFirstOrderLaplacian(f, v_bd_id, v)
            newv_newf = igl.remove_duplicates(v, f, epsilon=0.1)
            v, f = newv_newf[0], newv_newf[1]
        
        # refinedPatchV = refineMeshByFirstOrderLaplacian(patchF, patchV_id, v_bd_id, v)
        v_bd_id = igl.boundary_loop(f)
    v = igl.per_vertex_attribute_smoothing(v, f) # 整体 uniform laplacian smoothing
    # 对三角网格中较大的三角形进行上采样
    maxEdge = voxel_size # EDGE_SEP_COEF*avg_edge_length
    v, f = trimesh.remesh.subdivide_to_size(vertices=v, faces=f, max_edge=maxEdge, max_iter=10, return_index=False)
    # print(v.shape)

    # 生成voxel下采样点云
    downSampledVertices = voxelDownSample(v, voxel_size)
    np.savetxt(dstTxtFile, downSampledVertices)

    # 生成obj文件
    repairFaceWindingOrderAndNormals(v, f, dstObjFile)

    print("Finish Repairing {}".format(srcObjFile))





def getSrcDstFilePathList(srcObjRoot, dstObjRootDir, dstTxtRootDir, teethIndexRepairDirList):
    """Return: List of (srcObjFilePath, dstObjFilePath)"""
    if not os.path.exists(srcObjRoot):
        print("Source obj data does not exist.")
        return

    dstSubDirs = [os.path.join(dstObjRootDir, dir) for dir in teethIndexRepairDirList] + \
        [os.path.join(dstTxtRootDir, dir) for dir in teethIndexRepairDirList]
    for dstSubDir in dstSubDirs:
        if not os.path.exists(dstSubDir):
            os.makedirs(dstSubDir)
    
    firstSubDirFlag = True
    srcDstTupleList = []
    for root, dirs, fs in os.walk(srcObjRoot):
        if firstSubDirFlag: #第一级目录
            firstSubDirFlag = False
            continue
        else: # 第二级子目录
            subToothIndexDir = root.split("/")[-1]
            if subToothIndexDir not in teethIndexRepairDirList:
                continue
            for f in fs:
                fName, fType = tuple(f.split("."))
                if fType == "obj":
                    srcObjFile = os.path.join(root, f)
                    dstObjFile = os.path.join(dstObjRootDir, subToothIndexDir+"/"+fName+".obj")
                    dstTxtFile = os.path.join(dstTxtRootDir, subToothIndexDir+"/"+fName+".txt")
                    srcDstTupleList.append((srcObjFile,dstObjFile,dstTxtFile))
    return srcDstTupleList





if __name__ == "__main__":
    

    # srcObjRootDir = "./data/format-obj/"
    # dstObjRootDir = "./data/repaired-obj/"
    # dstTxtRootDir = "./data/repaired-txt/"

    # srcObjRootDir = "./data-refined/format-obj/"
    # dstObjRootDir = "./data-refined/repaired-obj/"
    # dstTxtRootDir = "./data-refined/repaired-txt/"

    srcObjRootDir = "./dataWithPhoto/format-obj/"
    dstObjRootDir = "./dataWithPhoto/repaired-obj/"
    dstTxtRootDir = "./dataWithPhoto/repaired-txt/"

    # teethIndexRepairList = ["11","12","21","22","13","14","23","24","33","34","43","44","15","16","17","25","26","27","35","36","37","45","46","47"] #["18","28","38","48"] 
    # FEASIBLE_COS_THRE = 0.6 # TOOTH 11---17,21---27

    teethIndexRepairList = ["31","32","41","42"]
    FEASIBLE_COS_THRE = 0.8 # TOOTH 31,32,41,42

    # # 将上一次生成的数据全部删除
    # # if os.path.exists(dstObjRootDir):
    # #     shutil.rmtree(dstObjRootDir)
    # # if os.path.exists(dstTxtRootDir):
    # #     shutil.rmtree(dstTxtRootDir)
    
    srcDstTupleList = getSrcDstFilePathList(srcObjRootDir, dstObjRootDir, dstTxtRootDir, teethIndexRepairList)
    keep_prev_flag = False # 是否保留上一次生成的数据
    voxel_size = 0.2 # 点云体素下采样的体素尺寸
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    # ray.get([generateOneRefinedPatchedTooth.remote(srcDstFs, keep_prev_flag, voxel_size) for srcDstFs in srcDstTupleList])
    
    # SINGLE TEST
    # fName = "11/63U.obj"
    # srcDstFs = (os.path.join(srcObjRootDir,fName), os.path.join(dstObjRootDir,fName), os.path.join(dstTxtRootDir,fName))
    # generateOneRefinedPatchedTooth(srcDstFs)
