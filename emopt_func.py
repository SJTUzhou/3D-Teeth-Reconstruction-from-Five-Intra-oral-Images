import numpy as np
import open3d as o3d
from shapely.geometry import MultiLineString, Point
from shapely.ops import unary_union, polygonize
import scipy
import cycpd
from scipy.spatial import Delaunay, distance_matrix
from scipy.spatial.transform import Rotation as RR
from collections import Counter
import itertools
import numba
import ray
import time
import math



################################
######### JIT functions ########
################################


@numba.njit(numba.float64[:,:](numba.float64[:,:], numba.float64[:,:]), cache=True)
def SquaredDistMatrix(x, y):
    n1 = len(x)
    n2 = len(y)
    d = x.shape[-1]
    distmat = np.zeros((n1,n2), np.float64)
    for i in range(n1):
        for j in range(n2):
            for k in range(d): 
                distmat[i,j] += (x[i,k] - y[j,k])**2
    return distmat

@numba.njit(numba.float64[:,:](numba.float64[:,:], numba.float64[:,:]), cache=True)
def DistMatrix(x, y):
    n1 = len(x)
    n2 = len(y)
    d = x.shape[-1]
    distmat = np.zeros((n1,n2), np.float64)
    for i in range(n1):
        for j in range(n2):
            for k in range(d): 
                distmat[i,j] += (x[i,k] - y[j,k])**2
            distmat[i,j] = math.sqrt(distmat[i,j])
    return distmat



def skewMatricesPy(a):
    # a: 2d-array, shape=(?,3)
    n, _ = a.shape
    vec_0 = np.zeros((n,1),np.float64)
    vec_a1, vec_a2, vec_a3 = np.split(a, 3, axis=-1)
    return np.stack([vec_0, -vec_a3, vec_a2, vec_a3, vec_0, -vec_a1, -vec_a2, vec_a1, vec_0], axis=-1).reshape((n,3,3))



@numba.njit(numba.float64[:,:,:](numba.float64[:,:]), cache=True)
def skewMatrices(a):
    # a: 2d-array, shape=(?,3)
    n = a.shape[0]
    ret = np.zeros((n,3,3),np.float64)
    for i in range(n):
        a1 = a[i,0]
        a2 = a[i,1]
        a3 = a[i,2]
        ret[i,0,1] = -a3
        ret[i,0,2] = a2
        ret[i,1,0] = a3
        ret[i,1,2] = -a1
        ret[i,2,0] = -a2
        ret[i,2,1] = a1
    return ret


    
@numba.njit(numba.float64[:,:](numba.float64[:,:],numba.intc), cache=True, parallel=True)
def farthestPointDownSample(vertices, num_point_sampled):
    # 最远点采样 FPS # vertices.shape = (N,3) or (N,2)
    N = vertices.shape[0]
    D = vertices.shape[1]
    assert num_point_sampled <= N, "Num of sampled point should be less than or equal to the size of vertices."
    # centroid of vertices
    _G = np.empty((D,),np.float64)
    for d in range(D):
        _G[d] = np.mean(vertices[:,d])

    dists = np.zeros((N,),np.float64)
    for i in numba.prange(N):
        for d in range(D):
            dists[i] += (vertices[i,d] - _G[d])**2
    farthest = np.argmax(dists) # 取离重心最远的点为起始点
    
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_) # 点是否被选中
    for _ in range(num_point_sampled):
        flags[farthest] = True
        distances[farthest] = 0.
        p_farthest = vertices[farthest]
        for i in numba.prange(N):
            dist = 0.
            if not flags[i]:
                for d in range(D):
                    dist += (vertices[i,d] - p_farthest[d])**2
            distances[i] = min(distances[i], dist)
        farthest = np.argmax(distances)
    return vertices[flags]





def initEdgeMaskNormals(vertices_xy, show=False):
    # 计算edgeMask ground truth中边缘点的法向量, shape = (M,2)
    M = len(vertices_xy)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.hstack([vertices_xy, 30*np.random.rand(M,1)]))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=30)
    normals_xy = np.asarray(pcd.normals)[:,:2]
    pcd.normals = o3d.utility.Vector3dVector(np.hstack([normals_xy, np.zeros((M,1))]))
    pcd.normalize_normals()
    if show == True:
        o3d.visualization.draw_geometries([pcd], window_name="image edge normals estimation", width=800, height=600, left=50,top=50, point_show_normal=True)
    return np.asarray(pcd.normals, dtype=np.float64)[:,:2]


def computeGroupedPointNormals(X): 
    # X.shape=(numGroup,numPoint,3)
    # 分别计算X中每组点云的法向量
    n = len(X)
    normals = np.empty(X.shape, np.float64)
    for i in range(n):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X[i])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
        # to obtain a consistent normal orientation
        # pcd.orient_normals_consistent_tangent_plane(k=30)
        pcd.normalize_normals()
        normals[i] = np.asarray(pcd.normals,dtype=np.float64)
    return normals















def solveCameraParamsbyDLT(p2d, p3d): 
    '''Pose Estimation by DLT'''
    assert len(p2d) == len(p3d), "Nums of 2D/3D points should be equal."
    N = len(p2d)
    pxl_x, pxl_y = np.split(p2d, indices_or_sections=2, axis=1) # 二维图像中的点的x,y坐标
    X = np.hstack([p3d, np.ones((N,1),np.float64)]) # shape=(N,4)
    O_Nx4 = np.zeros((N,4),np.float64)
    A = np.vstack( [np.hstack([X, O_Nx4, -pxl_x*X]), np.hstack([O_Nx4, X, -pxl_y*X])] )
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p_sol = vh[np.argmin(s),:]
    P = p_sol.reshape(3,4)
    _u, _s, _vh = np.linalg.svd(P, full_matrices=True)
    _c = _vh[-1,:]
    c = _c[:-1] / _c[-1]
    M = P[:,:3]
    Mt = P[:,3]
    R, Q = scipy.linalg.rq(M)
    t = -Q @ c
    F = R / R[-1,-1]
    assert np.all(np.diag(F)>0), "The diagonal values of R should be positive."
    assert np.allclose(R@Q, M), "RQ Decomposition Failed."
    return Q, t, F # return left-multiplying matrix


def icp_rigid_registration_2D(P_true, P_pred):
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(np.hstack([P_true,np.zeros((P_true.shape[0],1))]))
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(np.hstack([P_pred,np.zeros((P_pred.shape[0],1))]))
    
    s_init = np.prod(np.max(P_true,0)-np.min(P_true,0)) / np.prod(np.max(P_pred,0)-np.min(P_pred,0))
    txy = np.mean(P_true, 0) - np.mean(P_pred, 0)
    T0 = s_init * np.identity(4)
    T0[0,3] = txy[0]
    T0[1,3] = txy[1]
    T0[3,3] = 1.
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd_source, pcd_target, 1, T0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True))
    # print(reg_p2p.fitness, reg_p2p.inlier_rmse, len(reg_p2p.correspondence_set))
    _sR = reg_p2p.transformation[:2,:2].T
    _t = reg_p2p.transformation[:2,3]
    # print(reg_p2p.transformation)
    return _sR, _t


def loss_icp_rigid_registration_2D(P_true, P_pred, P_true_normals, P_pred_normals, varAngle):
    _sR, _t = icp_rigid_registration_2D(P_true, P_pred)
    TP_pred = P_pred @ _sR + _t
    TP_pred_normals = P_pred_normals @ _sR
    corre_pred_idx, e_losses = numba_get_global_point_correspondences(P_true, TP_pred, P_true_normals, TP_pred_normals, varAngle)
    return np.sum(e_losses), TP_pred





def cpd_rigid_registration_2D(P_true, P_pred):
    # rigid registration
    X = P_true.astype(np.double)
    Y = P_pred.astype(np.double)
    # 二维相似变换配准
    reg = cycpd.rigid_registration(**{'X': X, 'Y': Y, 'max_iterations':100,'tolerance':1.0,'w':1e-3,'verbose':False,'print_reg_params':False})
    TY,(s,r,t) = reg.register()
    return TY,s,r,t


@ray.remote
def ray_cpd_rigid_registration_2D(P_true, P_pred):
    # rigid registration
    X = P_true.astype(np.double)
    Y = P_pred.astype(np.double)
    # 二维相似变换配准
    reg = cycpd.rigid_registration(**{'X': X, 'Y': Y, 'max_iterations':100,'tolerance':1.0,'w':0.01,'verbose':False,'print_reg_params':False})
    TY,(s,r,t) = reg.register()
    return TY




@ray.remote
def loss_cpd_rigid_registration_2D(P_true, P_pred, P_true_normals, P_pred_normals, varAngle):
    TP_pred, s, r, t = cpd_rigid_registration_2D(P_true, P_pred)
    TP_pred_normals = s * P_pred_normals @ r
    corre_pred_idx, e_losses = get_global_point_correspondences(P_true, TP_pred, P_true_normals, TP_pred_normals, varAngle)
    return np.sum(e_losses), TP_pred



def rotvec2rotmat(rotVecXYZs):
    rotMats = RR.from_rotvec(rotVecXYZs).as_matrix()
    if rotVecXYZs.ndim == 1:
        return rotMats.T
    else:
        return np.transpose(rotMats, (0,2,1)) # 变为右乘




########################################################
# Isotropic scaled rigid transformation for each tooth #
########################################################

def updateTransformedPointPos(X_deformed, scales, rotMats, transVecXYZs, X_Mu_centroids):
    # X_trans = scales_inv * (X_deformed + transVecXYZs_inv) @ rotMats_inv
    # in CPD: X_aligned_deformed = scales_cpd * X_trans @ rotMats_cpd + transVecXYZs_cpd
    return np.multiply(scales[:,None,None], np.matmul(X_deformed-X_Mu_centroids[:,None,:], rotMats)) +\
            transVecXYZs[:,None,:] + X_Mu_centroids[:,None,:]

def updateTransformedPointNormals(X_deformed_normals, rotMats):
    # 法向量只需要计算旋转即可
    return np.matmul(X_deformed_normals, rotMats)




####################################################################
# Relative Pose of Lower Tooth Row with respect to Upper Tooth Row #
####################################################################

def updatePointPosByRelaPose(X, rela_R, rela_txyz, ul_sp):
    X_l = np.matmul(X[ul_sp:], rela_R) + rela_txyz
    return np.concatenate([X[:ul_sp], X_l])

def updatePointNormalsByRelaPose(X_normals, rela_R, ul_sp):
    X_ln = np.matmul(X_normals[ul_sp:], rela_R)
    return np.concatenate([X_normals[:ul_sp], X_ln])




###############################
# world coord -> camera coord #
###############################

def updateExtrinsicViewMatrix(ex_rxyz, ex_txyz): # world coord to camera coord
    # 先进行x轴旋转，再y轴，再z轴；取转置表示右乘旋转矩阵，再平移
    R = RR.from_rotvec(ex_rxyz).as_matrix().T
    return np.vstack([R, ex_txyz]) # Matrix 4*3
    
def updatePointPosInCameraCoord(X_world, extrViewMat):
    # get 3D point cloud in camera coord, return array shape (n,3) or (batch,n,3)
    X_homo = np.concatenate([X_world, np.ones((*X_world.shape[:-1],1))], axis=-1)
    return np.matmul(X_homo, extrViewMat)

def updatePointNormalsInCameraCoord(X_world_normals, extrViewRotMat):
    return np.matmul(X_world_normals, extrViewRotMat)



##############################
# camera coord ->image coord #
##############################

def updateIntrinsicProjectionMatrix(focLth, dpix, u0, v0): # camera cood to image coord
    return np.array([[focLth/dpix, 0., 0.], [0., focLth/dpix, 0.], [u0, v0, 1.]], np.float64)

def updatePointPosInImageCoord(X_camera, intrProjMat):
    # get 3D point cloud in image coord, return array shape (n,2) or (batch,n,2)
    # assert (X_camera[...,2]>0).all(), "max violation: {:.2f}".format(np.abs(np.min(X_camera[...,2]))) # Z-value of points should be positive
    invalid_z_val = X_camera[...,2] < 0
    X_camera[...,2][invalid_z_val] = 0.
    X_image = np.matmul((X_camera/X_camera[...,[2]]), intrProjMat)
    X_uv = X_image[...,:2]
    return X_uv

def updatePointNormalsInImageCoord(X_camera_normals):
    X_cam_normals_xy = X_camera_normals[...,:2] # 取相机坐标系下normals的x,y坐标即为图片坐标系中的normals
    return X_cam_normals_xy / np.linalg.norm(X_cam_normals_xy, axis=-1, keepdims=True)



##################################################
# Extract contour pixels in projected pointcloud #
##################################################

def getUniquePixels(X_uv_int):
    # merge points at the same position in image coord
    # X_uv_int: array shape (n,2)dtype np.int64
    # pixels: array (m,2), each element represents (u_x, v_y)
    pixels, unique_indices = np.unique(X_uv_int, axis=0, return_index=True)
    return pixels, unique_indices

def getConcaveHullEdgeVertexIndices(coords, alpha):  # coords is a 2D numpy array (u_x,v_y)
    tri = Delaunay(coords, qhull_options="Qt Qc Qz Q12").simplices
    ia, ib, ic = (tri[:, 0], tri[:, 1], tri[:, 2])  # indices of each of the triangles' points
    pa, pb, pc = (coords[ia], coords[ib], coords[ic])  # coordinates of each of the triangles' points
    a = np.linalg.norm(pa-pb, ord=2, axis=1)
    b = np.linalg.norm(pb-pc, ord=2, axis=1)
    c = np.linalg.norm(pc-pa, ord=2, axis=1)
    s = (a + b + c) * 0.5  # Semi-perimeter of triangle
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Area of triangle by Heron's formula
    filter = (a * b * c / (4.0 * area) < 1.0 / alpha)  # Radius Filter based on alpha value
    edges = tri[filter]
    edges = [tuple(sorted(combo)) for e in edges for combo in itertools.combinations(e, 2)]
    count = Counter(edges)  # count occurrences of each edge
    edge_indices = [e for e, c in count.items() if c == 1]
    return np.array(edge_indices)

def constructConcaveHull(coords, edge_indices): # coords is a 2D numpy array (u_x,v_y)
    edges = [(coords[e[0]], coords[e[1]]) for e in edge_indices]
    ml = MultiLineString(edges)
    poly = polygonize(ml)
    hull = unary_union(list(poly))
    return hull

def extractVisibleEdgePointsByAvgDepth(X_uv, priority):
    # X_uv: shape=(numTooth,numPoint,2), dtype=np.int64
    X_uv_sort = X_uv[priority]
    hulls = []
    vis_hull_vs = []
    vis_hull_vids = []
    for x_uv in X_uv_sort:
        pixels, pixel_xuv_map = getUniquePixels(x_uv)
        edge_v_indices = getConcaveHullEdgeVertexIndices(pixels, alpha=0.1)
        hull = constructConcaveHull(pixels, edge_v_indices)
        uni_edge_v_indices = np.unique(edge_v_indices)
        hull_v = x_uv[pixel_xuv_map[uni_edge_v_indices]]
        flags = np.ones((len(hull_v),),dtype=np.bool_)
        for i,v in enumerate(hull_v):
            for exist_hull in hulls:
                if exist_hull.contains(Point(v)):
                    flags[i] = False
                    break
        if flags.any()==True:
            hulls.append(hull)
            vis_hull_vs.append(hull_v[flags]) # 可见点
            vis_hull_vids.append(pixel_xuv_map[uni_edge_v_indices[flags]])
        else: #某颗牙齿的轮廓被完全遮住的情况
            vis_hull_vs.append(np.array([],dtype=hull_v.dtype).reshape((0,2)))  # empty point
            vis_hull_vids.append([])
    # sort in the init order
    vis_hull_vs = [x for _, x in sorted(zip(priority, vis_hull_vs), key=lambda pair: pair[0])]
    vis_hull_vids = [x for _, x in sorted(zip(priority, vis_hull_vids), key=lambda pair: pair[0])]
    return vis_hull_vs, vis_hull_vids





#################################################################
### VERSION 2: Extract contour pixels in projected pointcloud ###
#################################################################



@numba.njit(numba.types.Tuple((numba.int64[:],numba.float64[:],numba.float64[:]))(numba.float64[:,:], numba.intc, numba.double), cache=True)
def get_boundary_meta(pNx2, bincount=180, fluc=0.9):
    N = pNx2.shape[0]
    indices = - np.ones((bincount,), np.int64) # -1 denotes invalid
    center = np.array([np.mean(pNx2[:,0]), np.mean(pNx2[:,1])])
    pts = pNx2 - center
    theta = np.pi + np.arctan2(pts[:,1],pts[:,0]) # [0, 2PI]
    rho = np.sqrt(pts[:,1]**2 + pts[:,0]**2)
    rho_max = np.zeros((bincount,), np.float64)
    spc = 2 * np.pi / bincount
    for i in range(N):
        b = int(theta[i] / spc)
        if rho[i] > rho_max[b]:
            rho_max[b] = rho[i]
            indices[b] = i
    # 剔除不满足条件的边界点
    _rho_max = rho_max.copy()
    for b in range(-1,bincount-1):
        thresh = fluc * 0.5 * (_rho_max[b-1]+_rho_max[b+1])
        if rho_max[b] < thresh:
            indices[b] = -1
            rho_max[b] = thresh
    net_indices = indices[indices>=0] # valid boudary point indices
    return net_indices, center, rho_max


@numba.njit(numba.boolean[:](numba.float64[:,:], numba.float64[:], numba.float64[:]), cache=True)
def flag_out_of_region(pKx2, center, rho_max):
    K = pKx2.shape[0]    
    bincount = rho_max.shape[0]  
    flag = np.ones((K,), np.bool_)
    pts = pKx2 - center
    theta = np.pi + np.arctan2(pts[:,1],pts[:,0]) # [0, 2PI]
    rho = np.sqrt(pts[:,1]**2 + pts[:,0]**2)
    spc = 2 * np.pi / bincount
    for k in range(K):
        b = int(theta[k] / spc)
        if rho[k] < rho_max[b]:
            flag[k] = False
    return flag

@numba.njit(numba.types.Tuple((numba.types.List(numba.types.Array(numba.float64, 2, 'C')),numba.types.List(numba.types.Array(numba.int64, 1, 'C'))))\
            (numba.float64[:,:,:], numba.int64[:], numba.intc, numba.float64))
def extract_visible_edge_points(pMxNx2, priority, bincount=180, fluc=0.9):
    M = pMxNx2.shape[0]
    N = pMxNx2.shape[1]
    vis_hull_vs = []
    vis_hull_vids = []
    lst_bdry_pts = []
    lst_indices = []
    arr_centers = np.empty((M,2),np.float64)
    arr_rho_max = np.empty((M,bincount),np.float64)
    extents = np.empty((M,4),np.float64) # [xmin,xmax,ymin,ymax]
    
    b_ymin = int(0.25*bincount)-1
    b_ymax = int(0.75*bincount)-1
    b_xmin = 0
    b_xmax = int(0.5*bincount)-1
    for m in range(M):
        pNx2 = pMxNx2[m]
        _indices, _center, _rho_max = get_boundary_meta(pNx2, bincount, fluc)
        lst_indices.append(_indices)
        lst_bdry_pts.append(pNx2[_indices])
        arr_centers[m,0] = _center[0]
        arr_centers[m,1] = _center[1]
        arr_rho_max[m] = _rho_max
        extents[m,0] = _center[0] - _rho_max[b_xmin]
        extents[m,1] = _center[0] + _rho_max[b_xmax]
        extents[m,2] = _center[1] - _rho_max[b_ymin]
        extents[m,3] = _center[1] + _rho_max[b_ymax]        
    
    p_frontal = []
    for p in priority:
        pKx2 = lst_bdry_pts[p]
        Npt = pKx2.shape[0]
        mask = np.ones((Npt,),np.bool_)
        num_prev_p = len(p_frontal)
        if num_prev_p > 0:
            for i in range(num_prev_p):
                prev_p = p_frontal[i]
                if extents[p,0]>extents[prev_p,1] or extents[p,1]<extents[prev_p,0] or extents[p,2]>extents[prev_p,3] or extents[p,3]<extents[prev_p,2]:
                    continue
                flag = flag_out_of_region(pKx2, arr_centers[prev_p], arr_rho_max[prev_p])
                mask = np.logical_and(mask, flag)
        vis_hull_vs.append(pKx2[mask])
        vis_hull_vids.append(lst_indices[p][mask])
        p_frontal.append(p)
    
    # sort in the init order
    vis_hull_vs = [x for _, x in sorted(zip(priority, vis_hull_vs), key=lambda pair: pair[0])]
    vis_hull_vids = [x for _, x in sorted(zip(priority, vis_hull_vids), key=lambda pair: pair[0])]
    return vis_hull_vs, vis_hull_vids





##################################
######### Expectation Step #######
##################################

def get_global_point_correspondences(P_true, P_pred, P_true_normals, P_pred_normals, varAngle):
    Np = len(P_true)
    losses = np.zeros((Np,),np.float64)
    point_loss_mat = distance_matrix(P_true, P_pred, 2, 1e8) 
    normal_loss_mat = - (P_true_normals @ P_pred_normals.T)**2 / varAngle
    loss_mat = point_loss_mat * np.exp(normal_loss_mat) # weighted loss matrix
    corre_pred_idx = np.argmin(loss_mat, axis=1)
    for i in range(Np):
        losses[i] = loss_mat[i,corre_pred_idx[i]]
    return corre_pred_idx.astype(np.int64), losses






@numba.njit(numba.types.Tuple((numba.int64[:],numba.float64[:]))\
    (numba.types.Array(numba.float64, 2, 'C'), numba.types.Array(numba.float64, 2, 'C'), \
    numba.types.Array(numba.float64, 2, 'C'), numba.types.Array(numba.float64, 2, 'C'), numba.double))
def numba_get_global_point_correspondences(P_true, P_pred, P_true_normals, P_pred_normals, varAngle):
    Np = len(P_true)
    losses = np.zeros((Np,),np.float64)
    point_loss_mat = SquaredDistMatrix(P_true, P_pred)
    normal_loss_mat = - (P_true_normals @ P_pred_normals.T)**2 / varAngle
    loss_mat = point_loss_mat * np.exp(normal_loss_mat) # weighted loss matrix
    corre_pred_idx = np.argmin(loss_mat, axis=1)
    for i in range(Np):
        losses[i] = loss_mat[i,corre_pred_idx[i]]
    return corre_pred_idx.astype(np.int64), losses





import utils
if __name__ == "__main__":
    vertices = np.random.random((10000,2))
    tic = time.time()
    x1 = farthestPointDownSample(vertices, 4000)
    x2 = utils.farthestPointDownSample(vertices, 4000)
    toc = time.time()
    print(toc-tic)
    print(np.allclose(x1,x2))