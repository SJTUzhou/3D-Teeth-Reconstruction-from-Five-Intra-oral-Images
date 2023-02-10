import matplotlib as mpl
import open3d as o3d
import os
import copy
import numpy as np
import h5py
import scipy
import matplotlib.pyplot as plt
import skimage
from scipy.spatial.transform import Rotation as RR
import pcd_mesh_utils
from const import *




WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200





def generateProjectedMeshImg(visualizer, ulTeethMshes, phType, ex_rxyz, ex_txyz, fx, u0, v0, rela_R, rela_t, rh, rw):
    """set camera params, project open3d.geometry.TriangleMesh, and get 2d projection"""
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
    msh.rotate(rotMat, center=(0,0,0))
    msh.translate(ex_txyz[ph])
    visualizer.add_geometry(msh)
    viewControl = visualizer.get_view_control()
    pinholeParams = o3d.camera.PinholeCameraParameters()
    pinholeParams.intrinsic = o3d.camera.PinholeCameraIntrinsic(WINDOW_WIDTH, WINDOW_HEIGHT, fx[ph], fx[ph], u0[ph], v0[ph])  # 399.5, 299.5)
    pinholeParams.extrinsic = np.identity(4)
    viewControl.convert_from_pinhole_camera_parameters(pinholeParams, allow_arbitrary=True)
    visualizer.update_geometry(msh)
    visualizer.poll_events()
    visualizer.update_renderer()

    _u0 = WINDOW_WIDTH / 2 - 0.5
    _v0 = WINDOW_HEIGHT / 2 - 0.5
    img = np.asarray(visualizer.capture_screen_float_buffer(do_render=True))
    tForm = skimage.transform.EuclideanTransform(rotation=None, translation=np.array([ _u0-u0[ph], _v0-v0[ph]]), dimensionality=2)
    shiftedImg = skimage.transform.warp(img, tForm)
    croppedImg = shiftedImg[:rh, :rw]
    return croppedImg


def readCameraParamsFromH5(h5File):
    with h5py.File(h5File, 'r') as f:
        grp = f["EMOPT"]
        ex_rxyz = grp["EX_RXYZ"][:]
        ex_txyz = grp["EX_TXYZ"][:]
        focLth = grp["FOCLTH"][:]
        dpix = grp["DPIX"][:]
        u0 = grp["U0"][:]
        v0 = grp["V0"][:]
        rela_R = grp["RELA_R"][:]
        rela_t = grp["RELA_T"][:]
        return ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_R, rela_t
    

def meshProjection(visualizer, tag):
    """Overlay semi-transparent mesh projection on other images"""
    demoH5File = os.path.join(DEMO_H5_DIR, f"demo-tag={tag}.h5")
    upperTeethObj = os.path.join(DEMO_MESH_DIR, str(tag), r"Pred_Upper_Mesh_Tag={}.obj".format(tag))
    lowerTeethObj = os.path.join(DEMO_MESH_DIR, str(tag), r"Pred_Lower_Mesh_Tag={}.obj".format(tag))
    photos = []
    for phtype in PHOTO_TYPES:
        imgfile = os.path.join(PHOTO_DIR, f"{tag}-{phtype.value}.png")
        img = skimage.io.imread(imgfile)
        h, w = img.shape[:2]
        scale = RECONS_IMG_WIDTH / w
        rimg = skimage.transform.resize(img, (int(h*scale), RECONS_IMG_WIDTH, 3))
        photos.append(rimg)
        
    ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_R, rela_t = readCameraParamsFromH5(h5File=demoH5File)
    fx = focLth / dpix

    _color = [0.55, 0.7, 0.85]
    # _color = [0.75, 0.75, 0.75]
    # _color = [1., 0.843, 0.] # golden
    _alpha = 0.45

    upperTeethO3dMsh = o3d.io.read_triangle_mesh(upperTeethObj)
    upperTeethO3dMsh.paint_uniform_color(_color)
    upperTeethO3dMsh.compute_vertex_normals()

    lowerTeethO3dMsh = o3d.io.read_triangle_mesh(lowerTeethObj)
    lowerTeethO3dMsh.paint_uniform_color(_color)
    lowerTeethO3dMsh.compute_vertex_normals()

    for phType, img in zip(PHOTO_TYPES, photos):
        mshImg = generateProjectedMeshImg(visualizer, [upperTeethO3dMsh,lowerTeethO3dMsh], phType, ex_rxyz, ex_txyz, fx, u0, v0, rela_R, rela_t, img.shape[0], img.shape[1])
        mesh_img_file = os.path.join(VIS_DIR, f"mesh-tag={tag}-{phType}.png")
        skimage.io.imsave(mesh_img_file, skimage.img_as_ubyte(mshImg))
        
        bkgrd = np.all(mshImg < 0.01, axis=-1)
        _teethRegion = np.tile(~bkgrd[...,None],(1,1,3)) 
        img = img[...,:3]
        np.putmask(img, _teethRegion, np.clip(_alpha*mshImg+(1.-_alpha)*img, 0., 1.))
        output = img
        output_img_file = os.path.join(VIS_DIR, f"overlay-tag={tag}-{phType}.png")       
        skimage.io.imsave(output_img_file, skimage.img_as_ubyte(output))




def main():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Image Screen Shot", visible=True, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color # Normal

    for tag in ["0","1"]:
        meshProjection(vis, tag)
    vis.destroy_window()

    






if __name__ == "__main__":
    main()






        
    


