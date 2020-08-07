import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

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


useGazebo = False
nImages = 5
while(n < nImages):
		n = n+1


		color_raw = cv2.imread(str(n)+"_rgb.jpg",cv2.COLOR_RGB2BGR)
		depth_raw = cv2.imread(str(n)+"_depth.png",cv2.IMREAD_UNCHANGED)
		mask = cv2.imread(str(n)+"_mask.png",cv2.IMREAD_UNCHANGED)


		


		#mask = np.expand_dims(mask, 2)
		#mask_3c = np.append(mask, mask, 2)
		#mask_3c = np.append(mask_3c, mask, 2)

		masked = np.multiply(depth_raw, mask)

		print(color_raw.shape)


		plt.subplot(1, 3, 1)
		plt.title('Color')
		plt.imshow(color_raw)
		plt.subplot(1, 3, 2)
		plt.title('Depth')
		plt.imshow(depth_raw)
		plt.subplot(1, 3, 3)
		plt.title('Mask')
		plt.imshow(masked)
		plt.show()

		

		rgb_im = o3d.geometry.Image(color_raw)
		depth_im = o3d.geometry.Image(depth_raw)
		mask_im = o3d.geometry.Image(masked)


		pcd = open_pointCloud_from_rgb_and_depth(rgb_im, depth_im, meters_trunc=6, showImages=False)
		o3d.visualization.draw_geometries([pcd])


		pcd = open_pointCloud_from_rgb_and_depth(rgb_im, mask_im, meters_trunc=6, showImages=False)
		o3d.visualization.draw_geometries([pcd])


		# DBSCAN
		# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
		#     labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))

		# max_label = labels.max()
		# print(f"point cloud has {max_label + 1} clusters")
		# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
		# colors[labels < 0] = 0
		# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
		# o3d.visualization.draw_geometries([pcd])

		## Downsample
		# pcd = pcd.voxel_down_sample(voxel_size=0.05)
		# o3d.visualization.draw_geometries([pcd])

		# Voxelization
		# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)
		# ptmax = np.asarray(pcd.points).max(axis=0)
		# ptmin = np.asarray(pcd.points).min(axis=0)
		# eixo = [[0, 0, 0],[1, 0, 0],[0, 1, 0], [0, 0, 1]]
		# pts = np.array([ptmax, ptmin])
		# print(pts)
		# boundingGrid = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(eixo))
		# o3d.visualization.draw_geometries([voxel_grid, boundingGrid])

		## Remove radius outlier
		# cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
		# display_inlier_outlier(pcd, ind)
		# inlier_cloud = pcd.select_by_index(ind)
		# o3d.visualization.draw_geometries([inlier_cloud])

		# Statistical oulier removal
		# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
		# display_inlier_outlier(pcd, ind)
		# pcd = pcd.select_by_index(ind)
		# o3d.visualization.draw_geometries([pcd])

		# Ransac total
		# outlier_cloud = pcd
		# inlier_cloud_list = []
		# planosMaximos = 3

		# for x in range(planosMaximos):
		# 	# Ransac planar
		# 	plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.1,ransac_n=3,num_iterations=1000)
		# 	[a, b, c, d] = plane_model
		# 	print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
		# 	inlier_cloud_list.append(outlier_cloud.select_by_index(inliers).paint_uniform_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
		# 	outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
		# 	o3d.visualization.draw_geometries(inlier_cloud_list)

		# inlier_cloud_list.append(outlier_cloud)
		# o3d.visualization.draw_geometries(inlier_cloud_list)




		