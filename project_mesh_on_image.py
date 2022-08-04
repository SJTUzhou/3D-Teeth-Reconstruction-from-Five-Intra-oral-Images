import matplotlib as mpl
import open3d as o3d
import os
import copy
import numpy as np
import pandas as pd
import h5py
import scipy
import matplotlib.pyplot as plt
import skimage
from scipy.spatial.transform import Rotation as RR
import projection_utils as proj
import utils
from projection_utils import PHOTO, MASK_FRONTAL, MASK_LEFT, MASK_LOWER, MASK_RIGHT, MASK_UPPER



H5_DIR = r"./dataWithPhoto/demo/" # r"./dataWithPhoto/demo/Grad-99%conf-v21-PC=10/"
MESH_DIR = r"./dataWithPhoto/demoMesh/" # r"./dataWithPhoto/demoMesh/Grad-99%conf-v21-PC=10/"
PHOTO_DIR = r"./dataWithPhoto/normal_resized/"
EDGE_DIR = r"./dataWithPhoto/normal_mask/"
IMG_WIDTH = 800
IMG_HEIGHT = 600
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200
OUTPUT_DIR = r"./dataWithPhoto/photoWithMeshProjected/"
# OUTPUT_DIR = r"./dataWithPhoto/photoWithMeshProjected/Edge/"
NAME_IDX_MAP_CSV = r"./dataWithPhoto/nameIndexMapping.csv"
NAME_IDX_MAP = pd.read_csv(NAME_IDX_MAP_CSV)
PHOTO_ORDER = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
PHOTO_MASKS = [MASK_UPPER, MASK_LOWER, MASK_LEFT, MASK_RIGHT, MASK_FRONTAL]
PHOTO_TYPES = ["upperPhoto","lowerPhoto","leftPhoto","rightPhoto","frontalPhoto"]




def generateProjectedMeshImg(tagID, visualizer, ulTeethMshes, phType, ex_rxyz, ex_txyz, fx, u0, v0, rela_R, rela_t):
    """加载o3dTriangleMesh到指定的visualizer中,并设置相机参数进行投影,返回投影得到的2d图像的数组"""
    visualizer.clear_geometries()
    ph = phType.value
    msh = None
    if phType in [PHOTO.UPPER, PHOTO.LOWER]:
        msh = copy.deepcopy(ulTeethMshes[ph])
    else: # [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]]
        uMsh = copy.deepcopy(ulTeethMshes[0])
        lMsh = copy.deepcopy(ulTeethMshes[1])
        lMsh.rotate(rela_R.T, center=(0,0,0))
        lMsh.translate(rela_t)
        msh = uMsh + lMsh # merge meshes
    rotMat = RR.from_rotvec(ex_rxyz[ph]).as_matrix()
    # rotMat = RR.from_euler("xyz", ex_rxyz[ph]).as_matrix()
    msh.rotate(rotMat, center=(0,0,0))
    msh.translate(ex_txyz[ph])
    visualizer.add_geometry(msh)
    viewControl = visualizer.get_view_control()
    pinholeParams = o3d.camera.PinholeCameraParameters()
    pinholeParams.intrinsic = o3d.camera.PinholeCameraIntrinsic(WINDOW_WIDTH, WINDOW_HEIGHT, fx[ph], fx[ph], u0[ph], v0[ph])  # 399.5, 299.5)
    pinholeParams.extrinsic = np.identity(4)
    viewControl.convert_from_pinhole_camera_parameters(pinholeParams, allow_arbitrary=True)
    # camera_parameters = viewControl.convert_to_pinhole_camera_parameters()
    # print("Camera parameters\n{}\n{}".format(camera_parameters.extrinsic, camera_parameters.intrinsic.intrinsic_matrix))
    visualizer.update_geometry(msh)
    visualizer.poll_events()
    visualizer.update_renderer()

    _u0 = WINDOW_WIDTH / 2 - 0.5
    _v0 = WINDOW_HEIGHT / 2 - 0.5
    img = np.asarray(visualizer.capture_screen_float_buffer(do_render=True))
    tForm = skimage.transform.EuclideanTransform(rotation=None, translation=np.array([ _u0-u0[ph], _v0-v0[ph]]), dimensionality=2)
    shiftedImg = skimage.transform.warp(img, tForm)
    croppedImg = shiftedImg[:IMG_HEIGHT, :IMG_WIDTH]
    # print(croppedImg.shape)
    return croppedImg


def meshProjection(visualizer, tagID):
    """将所有的牙齿进行投影,并叠加到照片上"""
    demoH5File = os.path.join(H5_DIR, r"demo_TagID={}.h5".format(tagID))
    upperTeethObj = os.path.join(MESH_DIR, str(tagID), r"Pred_Upper_Mesh_TagID={}.obj".format(tagID))
    lowerTeethObj = os.path.join(MESH_DIR, str(tagID), r"Pred_Lower_Mesh_TagID={}.obj".format(tagID))
    ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_R, rela_t = proj.readCameraParamsFromH5(h5File=demoH5File, patientId=tagID)
    fx = focLth / dpix

    photos = proj.getPhotos(PHOTO_DIR, NAME_IDX_MAP, tagID, PHOTO_TYPES, (IMG_HEIGHT, IMG_WIDTH))

    _color = [0.55, 0.7, 0.85]
    _color = [0.75, 0.75, 0.75]
    _alpha = 0.45

    upperTeethO3dMsh = o3d.io.read_triangle_mesh(upperTeethObj)
    upperTeethO3dMsh.paint_uniform_color(_color)
    upperTeethO3dMsh.compute_vertex_normals()

    lowerTeethO3dMsh = o3d.io.read_triangle_mesh(lowerTeethObj)
    lowerTeethO3dMsh.paint_uniform_color(_color)
    lowerTeethO3dMsh.compute_vertex_normals()

    for phType, img in zip(PHOTO_ORDER, photos):
        mshImg = generateProjectedMeshImg(tagID, visualizer, [upperTeethO3dMsh,lowerTeethO3dMsh], phType, ex_rxyz, ex_txyz, fx, u0, v0, rela_R, rela_t)
        bkgrd = np.all(mshImg < 0.01, axis=-1)
        _teethRegion = np.tile(~bkgrd[...,None],(1,1,3)) 
        img = img[...,:3]
        np.putmask(img, _teethRegion, np.clip(_alpha*mshImg+(1.-_alpha)*img, 0., 1.))
        output = img
        output_img_file = os.path.join(OUTPUT_DIR, "{}-{}.png".format(tagID, str(phType)))
        print(output_img_file)
        skimage.io.imsave(output_img_file, skimage.img_as_ubyte(mshImg))
        # skimage.io.imsave(output_img_file, skimage.img_as_ubyte(output))










def meshProjectionWithSelectedTeeth(visualizer, tagID):
    """根据照片种类的不同对指定编号的牙齿进行投影"""
    demoH5File = os.path.join(H5_DIR, r"demo_TagID={}.h5".format(tagID))
    with h5py.File(demoH5File, 'r') as f:
        grp = f[str(tagID)]
        X_Pred_Upper = grp["UPPER_PRED"][:]
        X_Pred_Lower = grp["LOWER_PRED"][:]
        Mask = np.array(grp["MASK"][:], dtype=np.bool_)

    upperMeshList = [utils.surfaceVertices2WatertightO3dMesh(x) for x in X_Pred_Upper]
    lowerMeshList = [utils.surfaceVertices2WatertightO3dMesh(x) for x in X_Pred_Lower]
    numUpperT = len(upperMeshList)
    
    ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_R, rela_t = proj.readCameraParamsFromH5(h5File=demoH5File, patientId=tagID)
    fx = focLth / dpix

    photos = proj.getPhotos(PHOTO_DIR, NAME_IDX_MAP, tagID, PHOTO_TYPES, (IMG_HEIGHT, IMG_WIDTH))
    _color = [0.55, 0.7, 0.85]
    _color = [0.75, 0.75, 0.75]
    _alpha = 0.45

    for phType, phMask, img in zip(PHOTO_ORDER, PHOTO_MASKS, photos):
        visMask = phMask[Mask]
        upperTeethO3dMsh = utils.mergeO3dTriangleMeshes([_msh for _msh,_vm in zip(upperMeshList,visMask[:numUpperT]) if _vm==True])
        lowerTeethO3dMsh = utils.mergeO3dTriangleMeshes([_msh for _msh,_vm in zip(lowerMeshList,visMask[numUpperT:]) if _vm==True])
        if phType != PHOTO.LOWER:
            upperTeethO3dMsh.paint_uniform_color(_color)
            upperTeethO3dMsh.compute_vertex_normals()
        if phType != PHOTO.UPPER:
            lowerTeethO3dMsh.paint_uniform_color(_color)
            lowerTeethO3dMsh.compute_vertex_normals()
        
        mshImg = generateProjectedMeshImg(tagID, visualizer, [upperTeethO3dMsh,lowerTeethO3dMsh], phType, ex_rxyz, ex_txyz, fx, u0, v0, rela_R, rela_t)
        bkgrd = np.all(mshImg < 0.01, axis=-1)
        _teethRegion = np.tile(~bkgrd[...,None],(1,1,3)) 
        img = img[...,:3]
        np.putmask(img, _teethRegion, np.clip(_alpha*mshImg+(1.-_alpha)*img, 0., 1.))
        output = img
        output_img_file = os.path.join(OUTPUT_DIR, "{}-{}.png".format(tagID, str(phType)))
        print(output_img_file)
        # skimage.io.imsave(output_img_file, skimage.img_as_ubyte(output)) # project mesh on photos
        skimage.io.imsave(output_img_file, skimage.img_as_ubyte(mshImg)) # only project mesh









def vertex_error(x_pred, x_ref):
    """计算两组点云中点之间的最短距离"""
    dist_mat = scipy.spatial.distance_matrix(x_pred, x_ref, p=2, threshold=int(1e8))
    return np.min(dist_mat, axis=1)

def get_color_array(metric_array, metric_min=0., metric_max=1.8, plt_cmap_name="jet"):
    """generate color array based on the metric array; 
    the metric array is clipped by [metric_min, metric_max] and normalized to [0,1] """
    _metric = np.clip(metric_array, metric_min, metric_max)
    _metric = _metric / (metric_max-metric_min) # nor
    _cmap = plt.cm.get_cmap(plt_cmap_name)
    _colors = _cmap(_metric) # numpy.array, shape=(len(metric_array),4)
    return _colors[:,:3] # drop alpha channel


def meshErrorProjectionWithSelectedTeeth(visualizer, tagID):
    """根据照片种类的不同对指定编号的牙齿的配准误差进行投影"""
    demoH5File = os.path.join(H5_DIR, r"demo_TagID={}.h5".format(tagID))
    with h5py.File(demoH5File, 'r') as f:
        grp = f[str(tagID)]
        X_Pred_Upper = grp["UPPER_PRED"][:]
        X_Pred_Lower = grp["LOWER_PRED"][:]
        X_Ref_Upper = grp["UPPER_REF"][:]
        X_Ref_Lower = grp["LOWER_REF"][:]
        Mask = np.array(grp["MASK"][:], dtype=np.bool_)
    ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_R, rela_t = proj.readCameraParamsFromH5(h5File=demoH5File, patientId=tagID)
    fx = focLth / dpix

    with_scale = True
    T_Upper = utils.computeTransMatByCorres(X_Pred_Upper.reshape(-1,3), X_Ref_Upper.reshape(-1,3), with_scale=with_scale)
    T_Lower = utils.computeTransMatByCorres(X_Pred_Lower.reshape(-1,3), X_Ref_Lower.reshape(-1,3), with_scale=with_scale)

    upperMeshList = [utils.surfaceVertices2WatertightO3dMesh(x) for x in X_Pred_Upper]
    lowerMeshList = [utils.surfaceVertices2WatertightO3dMesh(x) for x in X_Pred_Lower]
    numUpperT = len(upperMeshList)

    upperColors = []
    lowerColors = []
    for i,_msh in enumerate(upperMeshList):
        _vertices = np.matmul(np.asarray(_msh.vertices), T_Upper[:3,:3]) + T_Upper[3,:3]
        _err = vertex_error(_vertices, X_Ref_Upper[i])
        upperColors.append(get_color_array(_err))
    for i,_msh in enumerate(lowerMeshList):
        _vertices = np.matmul(np.asarray(_msh.vertices), T_Lower[:3,:3]) + T_Lower[3,:3]
        _err = vertex_error(_vertices, X_Ref_Lower[i])
        lowerColors.append(get_color_array(_err))

    
    for phType, phMask in zip(PHOTO_ORDER, PHOTO_MASKS):
        visMask = phMask[Mask]
        upperTeethO3dMsh = utils.mergeO3dTriangleMeshes([_msh for _msh,_vm in zip(upperMeshList,visMask[:numUpperT]) if _vm==True])
        lowerTeethO3dMsh = utils.mergeO3dTriangleMeshes([_msh for _msh,_vm in zip(lowerMeshList,visMask[numUpperT:]) if _vm==True])
        if phType != PHOTO.LOWER:
            upperTeethO3dMsh.compute_vertex_normals()
            _colors = np.vstack([_c for _c,_m in zip(upperColors, visMask[:numUpperT]) if _m==True])
            upperTeethO3dMsh.vertex_colors = o3d.utility.Vector3dVector(_colors)

        if phType != PHOTO.UPPER:
            lowerTeethO3dMsh.compute_vertex_normals()
            _colors = np.vstack([_c for _c,_m in zip(lowerColors, visMask[numUpperT:]) if _m==True])
            lowerTeethO3dMsh.vertex_colors = o3d.utility.Vector3dVector(_colors)
        mshImg = generateProjectedMeshImg(tagID, visualizer, [upperTeethO3dMsh,lowerTeethO3dMsh], phType, ex_rxyz, ex_txyz, fx, u0, v0, rela_R, rela_t)
        output_img_file = os.path.join(OUTPUT_DIR, "{}-{}.png".format(tagID, str(phType)))
        print(output_img_file)
        skimage.io.imsave(output_img_file, skimage.img_as_ubyte(mshImg))




def color_bar():
    fig, ax = plt.subplots(figsize=(9,1.5))
    fig.subplots_adjust(bottom=0.5)

    cmap = mpl.cm.get_cmap("jet")
    norm = mpl.colors.Normalize(vmin=0, vmax=1.8)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb.set_label('Alignment Error (mm)',size=16)
    cb.ax.tick_params(labelsize=16)
    # fig.show()
    plt.show()



def main():
    TagIDRange = range(0,95) #[26,37,59,66] #range(0, 95)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Image Screen Shot", visible=True, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    # opt.background_color = np.asarray([1, 1, 1])
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color # Normal

    # vis.run() # block the visualizer
    for tagID in TagIDRange:
        # meshErrorProjectionWithSelectedTeeth(vis, tagID)
        # meshProjectionWithSelectedTeeth(vis, tagID)
        meshProjection(vis, tagID)
    vis.destroy_window()

    






if __name__ == "__main__":
    main()
    # color_bar()