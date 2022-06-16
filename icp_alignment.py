import numpy as np
import os
import open3d as o3d
import copy
import igl


""" 使用FPFH粗配准+ICP点到面精配准对两个牙齿点云进行配准
"""
# DEMO_OBJ_PATH = "./data/repaired-obj/22/"
DEMO_NPY_PATH = "./data/repaired-npy/11/"


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.5f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.5f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.5f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                               target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.5f," % voxel_size)
    print("   we use a liberal distance threshold %.5f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 6, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold)
    return result


def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.5f." % distance_threshold)
    radius_normal = 2*voxel_size
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def main():
    # objDataPath = DEMO_OBJ_PATH
    # pcdList = []
    # for f in os.listdir(objDataPath):
    #     objFPath = os.path.join(objDataPath, f)
    #     vertices, _, normals, faces, _, _ = igl.read_obj(objFPath)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(vertices)
    #     pcdList.append(pcd)

    npyDataPath = DEMO_NPY_PATH
    pcdList = []
    for f in os.listdir(npyDataPath):
        npyFPath = os.path.join(npyDataPath, f)
        vertices = np.load(npyFPath)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcdList.append(pcd)
    # o3d.visualization.draw_geometries(pcdList)
    
    

    source = np.random.choice(pcdList)
    target = np.random.choice(pcdList)
    srcMean, srcCovMat = source.compute_mean_and_covariance()
    targetMean, targetCovMat = target.compute_mean_and_covariance()
    print("src pcl mean: ", srcMean)
    print("target pcl mean: ", targetMean)
    print("src covariance matrix: ", srcCovMat)
    print("target covariance matrix: ", targetCovMat)
    trans_init = np.eye(4)
    # trans_init[0:3,3] = targetMean - srcMean
    # draw_registration_result(source, target, trans_init)

    # source = source.transform(trans_init)
    voxel_size = 0.8 # 1mm
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size) 
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    # draw_registration_result(source_down, target_down, np.identity(4))
    
    # print(np.asarray(source_down.points).shape)

    result_ransac = execute_global_registration(source_down, target_down,
                                           source_fpfh, target_fpfh,
                                           voxel_size)
    print("Transformation is:")
    print(result_ransac)
    print(result_ransac.transformation)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)
    print(np.asarray(source.points).shape)
    source = source.voxel_down_sample(0.2)
    target = target.voxel_down_sample(0.2)
    print(np.asarray(source.points).shape)
    result_icp = refine_registration(source, target, result_ransac, voxel_size)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)


    # threshold = 1e-4
    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, np.eye(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)
    

if __name__ == "__main__":
    main()