import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import glob
import copy 
import os
from aux.aux import *
from aux.LocalScene import LocalScene
from aux.plane import Plane
n = 0

#list_depth = ["selecionadas_wall_box/1_depth.png"]
#list_rgb = ["selecionadas_wall_box/1_rgb.jpg"]

showNormals = True

list_depth = sorted(glob.glob("selecionadas_wall_box/*_depth.png"))
list_rgb = sorted(glob.glob("selecionadas_wall_box/*.jpg"))

transformationList = [] # Should be n-1 images

for i in range(len(list_rgb)):
	color_raw = o3d.io.read_image(list_rgb[i])
	depth_raw = o3d.io.read_image(list_depth[i])
	pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = True)
	o3d.visualization.draw_geometries([pcd])

	ls = LocalScene(pcd)

	#Ransac total
	ls.findMainPlanes()
	ls.defineGroundNormal()
	ls.showMainPlanes()


	## Remove radius outlier
	# cl, ind = not_planes.remove_radius_outlier(nb_points=500, radius=0.1)
	# filtered_not_planes = not_planes.select_by_index(ind)
	# display_inlier_outlier(not_planes, ind)
	# o3d.visualization.draw_geometries([filtered_not_planes])

	# # Statistical oulier removal
	# # cl, ind = not_planes.remove_statistical_outlier(nb_neighbors=1000, std_ratio=2)
	# # #display_inlier_outlier(pcd, ind)
	# # filtered_not_planes = not_planes.select_by_index(ind)
	# # o3d.visualization.draw_geometries([filtered_not_planes])



	# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
	# 	labels = np.array(filtered_not_planes.cluster_dbscan(eps=0.07, min_points=200, print_progress=False))

	# max_label = labels.max()
	# print(f"point cloud has {max_label + 1} clusters")
	# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
	# colors[labels < 0] = 0
	# #filtered_not_planes.colors = o3d.utility.Vector3dVector(colors[:, :3])
	# o3d.visualization.draw_geometries([filtered_not_planes])
	# cluster_array = []
	# for n_cluster in range(max_label+1):
	# 	index_from_cluster = np.where(labels == n_cluster)[0]

	# 	cluster = filtered_not_planes.select_by_index( index_from_cluster.tolist())
	# 	cluster_qnt_points = np.asarray(cluster.points).shape[0]
	# 	if(cluster_qnt_points > 4000):
	# 		if(showNormals):
	# 			o3d.io.write_point_cloud("caixa6.ply", cluster)
	# 			downpcd = cluster.voxel_down_sample(voxel_size=0.05)
	# 			downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=500))
	# 			downpcd.orient_normals_towards_camera_location()
	# 			downpcd.normalize_normals()

	# 			normais = np.asarray(downpcd.normals)
	# 			cor = np.asarray(downpcd.colors)

	# 			print(cor.shape)

	# 			for i in range(normais.shape[0]):
	# 				#print(normais[i,:])
	# 				cor[i, :] = (np.abs(normais[i,0]),np.abs(normais[i,1]),np.abs(normais[i,2]))

	# 			downpcd.colors = o3d.utility.Vector3dVector(cor)
	# 			o3d.visualization.draw_geometries([downpcd])


	# 		cluster_array.append(cluster)
	# 		obb = cluster.get_axis_aligned_bounding_box()
	# 		obb2 = cluster.get_oriented_bounding_box()
	# 		obb.color = (1,0,0)
	# 		obb2.color = (0,1,0)
	# 		cluster_array.append(obb)
	# 		cluster_array.append(obb2)
	
	# all_objects = inlier_cloud_list[0:-1]
	# all_objects.extend(cluster_array)
	# print(all_objects)
	# o3d.visualization.draw_geometries(all_objects)



