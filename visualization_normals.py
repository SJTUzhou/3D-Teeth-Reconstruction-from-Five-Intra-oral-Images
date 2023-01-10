import numpy as np
import open3d as o3d
import projection_utils as proj

def create_smartee_label_colormap():
    """Creates a label colormap used in Smartee tooth dataset.

    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [50, 50, 50]
    colormap[1] = [105, 105, 105]
    colormap[2] = [34, 139, 34]
    colormap[3] = [128, 0, 0]
    colormap[4] = [128, 128, 0]
    colormap[5] = [72, 61, 139]
    colormap[6] = [0, 128, 128]
    colormap[7] = [70, 130, 180]
    colormap[8] = [154, 205, 50]
    colormap[9] = [0, 0, 139]
    colormap[10] = [127, 0, 127]
    colormap[11] = [143, 188, 143]
    colormap[12] = [176, 48, 96]
    colormap[13] = [72, 209, 204]
    colormap[14] = [255, 0, 0]
    colormap[15] = [255, 165, 0]
    colormap[16] = [255, 255, 0]
    colormap[17] = [127, 255, 0]
    colormap[18] = [138, 43, 226]
    colormap[19] = [0, 255, 127]
    colormap[20] = [244, 164, 96]
    colormap[21] = [0, 0, 255]
    colormap[22] = [240, 128, 128]
    colormap[23] = [255, 99, 71]
    colormap[24] = [176, 196, 222]
    colormap[25] = [255, 0, 255]
    colormap[26] = [30, 144, 255]
    colormap[27] = [144, 238, 144]
    colormap[28] = [255, 20, 147]
    colormap[29] = [123, 104, 238]
    colormap[30] = [245, 222, 179]
    colormap[31] = [238, 130, 238]
    colormap[32] = [255, 192, 203]
    
    return colormap



def generatePcdWithNormals(vertices, name, pointcolor=np.array([1.,0.,0.]), show=True):
    # X.shape=(numPoint,3)
    center = np.mean(vertices, 0)
    vecs = vertices - center
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals()
    # to obtain a consistent normal orientation
    pcd.orient_normals_consistent_tangent_plane(k=30)
    pcd.normalize_normals()
    normal = np.asarray(pcd.normals)
    opposite_mask = np.sum(vecs * normal, 1) < 0
    normal[opposite_mask] = -normal[opposite_mask]
    pcd.normals = o3d.utility.Vector3dVector(normal)
    pcd.paint_uniform_color(pointcolor)
    if show == True:
        o3d.visualization.draw_geometries([pcd], window_name=name, width=800, height=600, left=50,top=50, point_show_normal=True)
    return pcd
        


SSM_DIR = r"./data/cpdGpAlignedData/eigValVec/" 

if __name__ == "__main__":
    colormap = create_smartee_label_colormap() / 255.
    colormap = colormap[2:]
    Mu, SqrtEigVals, Sigma = proj.loadMuEigValSigma(SSM_DIR, numPC=10)
    upper_pcds = []
    lower_pcds = []
    for i in range(len(Mu)):
        pcd = generatePcdWithNormals(Mu[i], f"{(i+1)//7+1}{(i+1)%7}",colormap[i],show=False)
        if i < len(Mu) // 2:
            upper_pcds.append(pcd)
        else:
            lower_pcds.append(pcd)
    
    o3d.visualization.draw_geometries(upper_pcds, window_name="", width=800, height=600, left=50,top=50, point_show_normal=False)
    o3d.visualization.draw_geometries(lower_pcds, window_name="", width=800, height=600, left=50,top=50, point_show_normal=False)