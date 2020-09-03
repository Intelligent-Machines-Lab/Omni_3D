import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import glob
import copy 
import os

n = 0




def open_pointCloud_from_files(n = 1, folder="images_a_gazebo", end_color = "_rgb.png", end_depth="_depth.png", meters_trunc = 6, showImages = True):
	depth_scale=1/1000
	color_raw = o3d.io.read_image(folder+"/"+str(n)+end_color)
	depth_raw = o3d.io.read_image(folder+"/"+str(n)+end_depth)

	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1/depth_scale, depth_trunc=meters_trunc, convert_rgb_to_intensity=False)
	print(rgbd_image)

	
	if(showImages):
		plt.subplot(1, 2, 1)
		plt.title('Redwood grayscale image')
		plt.imshow(rgbd_image.color)
		plt.subplot(1, 2, 2)
		plt.title('Redwood depth image')
		plt.imshow(rgbd_image.depth)
		plt.show()

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	return pcd



def open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=5, showImages = True):
	depth_scale=1/1000

	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1/depth_scale, depth_trunc=meters_trunc, convert_rgb_to_intensity=False)

	if(showImages):
		plt.subplot(1, 2, 1)
		plt.title('Redwood grayscale image')
		plt.imshow(rgbd_image.color)
		plt.subplot(1, 2, 2)
		plt.title('Redwood depth image')
		plt.imshow(rgbd_image.depth)
		plt.show()

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	return pcd



def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])








# color_raw = o3d.io.read_image("images_alinhadas/teste2_rgb.jpg")
# depth_raw = o3d.io.read_image("images_alinhadas/teste2_depth_raw.png")
# depth_pos = o3d.io.read_image("images_alinhadas/teste2_depth.png")
# pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=5, showImages=False)
# o3d.visualization.draw_geometries([pcd])
# pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_pos, meters_trunc=5, showImages=False)
# o3d.visualization.draw_geometries([pcd])

#list_depth = ["selecionadas_wall_box/10_depth.png"]
#list_rgb = ["selecionadas_wall_box/10_rgb.jpg"]

showNormals = True

list_depth = sorted(glob.glob("images_alinhadas/*_depth.png"))
list_rgb = sorted(glob.glob("images_alinhadas/*.jpg"))

transformationList = [] # Should be n-1 images

for i in range(len(list_rgb)):
	color_raw = o3d.io.read_image(list_rgb[i])
	depth_raw = o3d.io.read_image(list_depth[i])
	pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=4, showImages = True)
	o3d.visualization.draw_geometries([pcd])

	# downpcd = pcd.voxel_down_sample(voxel_size=0.05)
	# downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
	# downpcd.orient_normals_towards_camera_location()
	# downpcd.normalize_normals()
	# o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

	print('voxelization')
	voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)
	o3d.visualization.draw_geometries([voxel_grid])