import open3d as o3d
import os
import copy
import numpy as np
import pandas as pd
import open3d.visualization.rendering as rendering
import open3d.visualization as visualization
import skimage
from scipy.spatial.transform import Rotation as RR
import projection_utils as proj
from projection_utils import PHOTO



H5_DIR = r"./dataWithPhoto/demo/"
MESH_DIR = r"./dataWithPhoto/demoMesh/"
PHOTO_DIR = r"./dataWithPhoto/normal_resized/"
EDGE_DIR = r"./dataWithPhoto/normal_mask/"
IMG_WIDTH = 800
IMG_HEIGHT = 600
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200
# OUTPUT_DIR = r"./dataWithPhoto/photoWithMeshProjected/"
OUTPUT_DIR = r"./dataWithPhoto/photoWithMeshProjected/Edge/"
NAME_IDX_MAP_CSV = r"./dataWithPhoto/nameIndexMapping.csv"
NAME_IDX_MAP = pd.read_csv(NAME_IDX_MAP_CSV)
PHOTO_ORDER = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
PHOTO_TYPES = ["upperPhoto","lowerPhoto","leftPhoto","rightPhoto","frontalPhoto"]


def proj_msh_on_img(img, o3dMsh, fx, fy, cx, cy):
    img_height, img_width = img.shape[:2]
    # Create a renderer with a set image width and height
    render = rendering.OffscreenRenderer(img_width, img_height)
    # setup camera intrinsic values
    pinhole = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)
    # Pick a background colour of the rendered image, I set it as black (default is light gray)
    render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
    # (The base color does not replace the mesh's own colors.)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "normals" # "defaultLitTransparency" # "normals" # "defaultLit"
    # add mesh to the scene
    render.scene.add_geometry("Mesh Model", o3dMsh, mtl)
    # render the scene with respect to the camera
    # open3d 默认相机坐标系 +X points to the right; +Y points up; +Z points out of the screen
    # Sets the position and orientation of the camera: look_at(center(相机镜头朝向), eye(相机位置), up(相机up-vector))
    _center = np.array([0,0,1])
    _eye = np.array([0,0,0])
    _up = np.array([0,-1,0])
    render.scene.camera.look_at(_center, _eye, _up) # 将相机+X，+Y方向变为与图片坐标系一致
    render.scene.camera.set_projection(pinhole.intrinsic_matrix, 0.0, np.inf, img_width, img_height)
    # print(pinhole.intrinsic_matrix)
    # print(render.scene.camera.get_view_matrix()) # view matrix
    # print(render.scene.camera.get_model_matrix()) # model matrix 
    # print(render.scene.camera.get_projection_matrix()) # projection matrix
    img_o3d = render.render_to_image()
    # we can now save the rendered image right at this point 
    o3d.io.write_image("output.png", img_o3d, 9)




def generateProjectedMeshImg(tagID, visualizer, ulTeethMshes, phType, ex_rxyz, ex_txyz, fx, u0, v0, rela_R, rela_t):
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
    rotMat = RR.from_euler("xyz", ex_rxyz[ph]).as_matrix()
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
    demoH5File = os.path.join(H5_DIR, r"demo_TagID={}.h5".format(tagID))
    upperTeethObj = os.path.join(MESH_DIR, str(tagID), r"Pred_Upper_Mesh_TagID={}.obj".format(tagID))
    lowerTeethObj = os.path.join(MESH_DIR, str(tagID), r"Pred_Lower_Mesh_TagID={}.obj".format(tagID))
    ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_R, rela_t = proj.readCameraParamsFromH5(h5File=demoH5File, patientId=tagID)
    fx = focLth / dpix

    # photos = proj.getPhotos(PHOTO_DIR, NAME_IDX_MAP, tagID, PHOTO_TYPES, (IMG_HEIGHT, IMG_WIDTH))
    photos = proj.getPhotos(EDGE_DIR, NAME_IDX_MAP, tagID, PHOTO_TYPES, (IMG_HEIGHT, IMG_WIDTH))

    upperTeethO3dMsh = o3d.io.read_triangle_mesh(upperTeethObj)
    upperTeethO3dMsh.paint_uniform_color([0.75, 0.75, 0.75])
    upperTeethO3dMsh.compute_vertex_normals()

    lowerTeethO3dMsh = o3d.io.read_triangle_mesh(lowerTeethObj)
    lowerTeethO3dMsh.paint_uniform_color([0.8, 0.8, 0.8])
    lowerTeethO3dMsh.compute_vertex_normals()

    for phType, img in zip(PHOTO_ORDER, photos):
        mshImg = generateProjectedMeshImg(tagID, visualizer, [upperTeethO3dMsh,lowerTeethO3dMsh], phType, ex_rxyz, ex_txyz, fx, u0, v0, rela_R, rela_t)
        _mask = mshImg > 0
        img = img[...,:3]
        np.putmask(img, _mask, np.clip(0.4*mshImg+0.6*img, 0., 1.))
        output = img
        output_img_file = os.path.join(OUTPUT_DIR, "{}-{}.png".format(tagID, str(phType)))
        print(output_img_file)
        skimage.io.imsave(output_img_file, skimage.img_as_ubyte(output))




def main():
    TagIDRange = range(0, 95)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Image Screen Shot", visible=True, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    # vis.run() # block the visualizer
    for tagID in TagIDRange:
        meshProjection(vis, tagID)
    vis.destroy_window()

    






if __name__ == "__main__":
    main()