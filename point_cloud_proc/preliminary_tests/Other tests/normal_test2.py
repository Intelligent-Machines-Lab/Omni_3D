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





list_depth = sorted(glob.glob("selecionadas_wall_box/*_depth.png"), key=os.path.getmtime)
list_rgb = sorted(glob.glob("selecionadas_wall_box/*.jpg"), key=os.path.getmtime)

transformationList = [] # Should be n-1 images

for i in range(len(list_rgb)):
	color_raw = o3d.io.read_image(list_rgb[i])
	depth_raw = o3d.io.read_image(list_depth[i])


	img = np.asarray(color_raw)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.medianBlur(gray, 3)
	edged=cv2.Canny(gray,20,160)

	contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	mask = np.zeros(img.shape, np.uint8)
	cv.drawContours(mask, contours, -1, (0,255,0),1)



	plt.subplot(1, 2, 1)
	plt.title('Original')
	plt.imshow(img)
	plt.subplot(1, 2, 2)
	plt.title('Processada')
	plt.imshow(edged)
	plt.show()


	pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = False)

