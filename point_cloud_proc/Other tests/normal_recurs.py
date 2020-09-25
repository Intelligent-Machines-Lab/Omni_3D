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

list_depth = ["selecionadas_wall_box/6_depth.png"]
list_rgb = ["selecionadas_wall_box/6_rgb.jpg"]

showNormals = True

#list_depth = sorted(glob.glob("selecionadas_wall_box/*_depth.png"))
#list_rgb = sorted(glob.glob("selecionadas_wall_box/*.jpg"))

transformationList = [] # Should be n-1 images

for i in range(len(list_rgb)):
	color_raw = o3d.io.read_image(list_rgb[i])
	depth_raw = o3d.io.read_image(list_depth[i])
	pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = True)
	o3d.visualization.draw_geometries([pcd])



	#Ransac total
	outlier_cloud = pcd
	inlier_cloud_list = []
	planosMaximos = 3
	qtn_inliers = 99999999

	while(True):
		# Ransac planar
		plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05,ransac_n=3,num_iterations=1000)
		qtn_inliers = np.asarray(inliers).shape[0]
		if(qtn_inliers < 4000):
			break
		[a, b, c, d] = plane_model
		print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
		inlier_cloud_list.append(outlier_cloud.select_by_index(inliers).paint_uniform_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
		o3d.visualization.draw_geometries(inlier_cloud_list)

		outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
		cl, ind = outlier_cloud.remove_radius_outlier(nb_points=500, radius=0.1)
		display_inlier_outlier(outlier_cloud, ind)
		outlier_cloud = outlier_cloud.select_by_index(ind)

		



	inlier_cloud_list.append(outlier_cloud)
	o3d.visualization.draw_geometries([inlier_cloud_list[-1]])

	not_planes = inlier_cloud_list[-1]
