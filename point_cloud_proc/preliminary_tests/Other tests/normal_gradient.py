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

list_depth = ["selecionadas_wall_box/1_depth.png"]
list_rgb = ["selecionadas_wall_box/1_rgb.jpg"]

showNormals = True

#list_depth = sorted(glob.glob("selecionadas_wall_box/*_depth.png"))
#list_rgb = sorted(glob.glob("selecionadas_wall_box/*.jpg"))

transformationList = [] # Should be n-1 images

for i in range(len(list_rgb)):
	color_raw = o3d.io.read_image(list_rgb[i])
	depth_raw = o3d.io.read_image(list_depth[i])
	pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = True)
	o3d.visualization.draw_geometries([pcd])

	# downpcd = pcd.voxel_down_sample(voxel_size=0.05)
	# downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
	# downpcd.orient_normals_towards_camera_location()
	# downpcd.normalize_normals()
	# o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

	# print("Print the normal vectors of the first 10 points")
	# normais = np.asarray(downpcd.normals)
	# cor = np.asarray(downpcd.colors)
	# print(normais.shape)
	# print(cor.shape)

	# for i in range(normais.shape[0]):
	# 	#print(normais[i,:])
	# 	cor[i, :] = (np.abs(normais[i,0]),np.abs(normais[i,1]),np.abs(normais[i,2]))

	# downpcd.colors = o3d.utility.Vector3dVector(cor)
	# o3d.visualization.draw_geometries([downpcd])

	#Ransac total
	outlier_cloud = pcd
	inlier_cloud_list = []
	planosMaximos = 3
	qtn_inliers = 99999999

	while(True):
		# Ransac planar
		plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05,ransac_n=3,num_iterations=1000)
		qtn_inliers = np.asarray(inliers).shape[0]
		if(qtn_inliers < 40000):
			break
		[a, b, c, d] = plane_model
		print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
		inlier_cloud_list.append(outlier_cloud.select_by_index(inliers).paint_uniform_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
		outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
		o3d.visualization.draw_geometries(inlier_cloud_list)

	inlier_cloud_list.append(outlier_cloud)
	o3d.visualization.draw_geometries([inlier_cloud_list[-1]])

	not_planes = inlier_cloud_list[-1]


	## Remove radius outlier
	cl, ind = not_planes.remove_radius_outlier(nb_points=500, radius=0.1)
	filtered_not_planes = not_planes.select_by_index(ind)
	display_inlier_outlier(not_planes, ind)
	o3d.visualization.draw_geometries([filtered_not_planes])

	# Statistical oulier removal
	# cl, ind = not_planes.remove_statistical_outlier(nb_neighbors=1000, std_ratio=2)
	# #display_inlier_outlier(pcd, ind)
	# filtered_not_planes = not_planes.select_by_index(ind)
	# o3d.visualization.draw_geometries([filtered_not_planes])



	with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
		labels = np.array(filtered_not_planes.cluster_dbscan(eps=0.07, min_points=200, print_progress=False))

	max_label = labels.max()
	print(f"point cloud has {max_label + 1} clusters")
	colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
	colors[labels < 0] = 0
	filtered_not_planes.colors = o3d.utility.Vector3dVector(colors[:, :3])
	o3d.visualization.draw_geometries([filtered_not_planes])
	cluster_array = []
	for n_cluster in range(max_label+1):
		index_from_cluster = np.where(labels == n_cluster)[0]

		cluster = filtered_not_planes.select_by_index( index_from_cluster.tolist())
		cluster_qnt_points = np.asarray(cluster.points).shape[0]
		if(cluster_qnt_points > 1500):
			if(showNormals):
				downpcd = cluster#.voxel_down_sample(voxel_size=0.05)
				downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=500))
				downpcd.orient_normals_towards_camera_location()
				downpcd.normalize_normals()




				normais = np.asarray(downpcd.normals)
				cor = np.asarray(downpcd.colors)
				downpcd_tree = o3d.geometry.KDTreeFlann(downpcd)

				print(cor.shape)
				normal_diff = []

				for i in range(normais.shape[0]):
					#print(normais[i,:])
					cor[i, :] = (np.abs(normais[i,0]),np.abs(normais[i,1]),np.abs(normais[i,2]))

					
					[k, idx, _] = downpcd_tree.search_radius_vector_3d(downpcd.points[i], 0.05)
					normal_diff.append(np.sum(np.abs(normais[i, :] - normais[idx[1:], :])))
					#print(normal_diff[-1])
						
				normal_diff_max = np.max(normal_diff)
				print("max: " + str(normal_diff_max))
				normal_diff_normalized = normal_diff/normal_diff_max
				for i in range(normais.shape[0]):
					cor[i, :] = (normal_diff_normalized[i],0,0)
				downpcd.colors = o3d.utility.Vector3dVector(cor)
				o3d.visualization.draw_geometries([downpcd])


			cluster_array.append(cluster)
			#obb = cluster.get_axis_aligned_bounding_box()
			obb = cluster.get_oriented_bounding_box()
			obb.color = (random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1))
			cluster_array.append(obb)
	
	all_objects = inlier_cloud_list[0:-1]
	all_objects.extend(cluster_array)
	print(all_objects)
	o3d.visualization.draw_geometries(all_objects)



