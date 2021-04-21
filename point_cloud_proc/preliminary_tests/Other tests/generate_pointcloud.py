import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
n = 1
nImages = 1

depth_scale = 0.001
clipping_distance_in_cm_max = 30 #1 meter
maxvalue = 0

while(n <= nImages):
		n = n+1

		# dp = cv2.imread('images_alinhadas/1_depth.png',cv2.IMREAD_UNCHANGED)
		# depth_raw = o3d.geometry.Image(dp)

		color_raw = o3d.io.read_image("images_alinhadas/6_rgb.jpg")
		depth_raw = o3d.io.read_image('images_alinhadas/6_depth.png')

		rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1/depth_scale, depth_trunc=clipping_distance_in_cm_max, convert_rgb_to_intensity=False)
		print(rgbd_image)

		

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
		o3d.visualization.draw_geometries([pcd])


		