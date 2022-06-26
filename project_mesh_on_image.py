import open3d as o3d
import os
import numpy as np
import open3d.visualization.rendering as rendering
import open3d.visualization as visualization
import skimage
from scipy.spatial.transform import Rotation as RR
import projection_utils as proj



H5_DIR = r"./"
MESH_DIR = r"./"
IMG_DIR = r"./image/"
TAG_ID = 63
DEMO_H5_FILE = os.path.join(H5_DIR, r"demo_TagID={}.h5".format(TAG_ID))
UPPER_TEETH_MESH_OBJ = os.path.join(MESH_DIR, r"Aligned_Pred_Upper_Mesh_TagID={}.obj".format(TAG_ID))
LOWER_TEETH_MESH_OBJ = os.path.join(MESH_DIR, r"Aligned_Pred_Lower_Mesh_TagID={}.obj".format(TAG_ID))
IMG_FILE = os.path.join(IMG_DIR, r"安然_219474_下牙列.png")
IMG_WIDTH = 800
IMG_HEIGHT = 600



def proj_msh_on_img(img, o3dMsh, fx, fy, cx, cy):

    img_height, img_width = img.shape[:2]
    # Create a renderer with a set image width and height
    render = rendering.OffscreenRenderer(img_width, img_height)

    # setup camera intrinsic values
    pinhole = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)
        
    # Pick a background colour of the rendered image, I set it as black (default is light gray)
    render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA

    # Define a simple unlit Material.
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



def main():
    ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_R, rela_t = proj.readCameraParamsFromH5(h5File=DEMO_H5_FILE, patientId=TAG_ID)
    imgUpper = skimage.io.imread(IMG_FILE)
    imgUpper = skimage.transform.resize(imgUpper, (IMG_HEIGHT, IMG_WIDTH, 3), anti_aliasing=True)
    print(UPPER_TEETH_MESH_OBJ)
    upperTeethO3dMsh = o3d.io.read_triangle_mesh(LOWER_TEETH_MESH_OBJ)
    upperTeethO3dMsh.paint_uniform_color([0.8, 0.8, 0.8])
    idx = 1
    f = focLth / dpix
    rotMat = RR.from_euler("xyz", ex_rxyz[idx]).as_matrix()
    upperTeethO3dMsh.rotate(rotMat, center=(0,0,0))
    upperTeethO3dMsh.translate(ex_txyz[idx])
    upperTeethO3dMsh.compute_vertex_normals()

    # _vertices = np.asarray(upperTeethO3dMsh.vertices)
    # print(np.max(_vertices,axis=0))
    # print(np.min(_vertices,axis=0))
    # o3d.visualization.draw_geometries([upperTeethO3dMsh,], window_name="", width=800, height=600, left=50,top=50)
    # proj_msh_on_img(imgUpper, upperTeethO3dMsh, fx=f[idx], fy=f[idx], cx=u0[idx], cy=v0[idx])


    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="", visible=True, width=IMG_WIDTH, height=IMG_HEIGHT)
    vis.add_geometry(upperTeethO3dMsh)

    viewControl = vis.get_view_control()
    # viewControl.set_front([0,0,-1])
    # viewControl.set_lookat([0,0,0])
    # viewControl.set_up([0,-1,0])
    pinholeParams = o3d.camera.PinholeCameraParameters()
    pinholeParams.intrinsic = o3d.camera.PinholeCameraIntrinsic(IMG_WIDTH, IMG_HEIGHT, f[idx], f[idx], u0[idx], v0[idx])  # 399.5, 299.5)
    pinholeParams.extrinsic = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    viewControl.convert_from_pinhole_camera_parameters(pinholeParams, allow_arbitrary=True)


    camera_parameters = viewControl.convert_to_pinhole_camera_parameters()
    print("Camera parameters\n{}\n{}".format(camera_parameters.extrinsic, camera_parameters.intrinsic.intrinsic_matrix))
    

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()

    # vis.capture_screen_image("output.png", do_render=True)
    o3dImg = vis.capture_screen_float_buffer(do_render=True)
    _u0 = IMG_WIDTH / 2 - 0.5
    _v0 = IMG_HEIGHT / 2 - 0.5
    img = np.asarray(o3dImg)
    tForm = skimage.transform.EuclideanTransform(rotation=None, translation=np.array([ _u0-u0[idx], _v0-v0[idx]]), dimensionality=2)
    shiftedImg = skimage.transform.warp(img, tForm)
    skimage.io.imsave("output.png", shiftedImg)
    vis.destroy_window()


    # save image 



    # image translation

    







if __name__ == "__main__":
    main()