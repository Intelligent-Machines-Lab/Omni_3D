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

list_depth = sorted(glob.glob("selecionadas_wall_cylinder/*_depth.png"))
list_rgb = sorted(glob.glob("selecionadas_wall_cylinder/*.jpg"))

transformationList = [] # Should be n-1 images

for i in range(len(list_rgb)):
	color_raw = o3d.io.read_image(list_rgb[i])
	depth_raw = o3d.io.read_image(list_depth[i])
	pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = False)
	#o3d.visualization.draw_geometries([pcd])

	ls = LocalScene(pcd)

	#Ransac total
	ls.findMainPlanes()
	ls.defineGroundNormal()
	o3d.visualization.draw_geometries(ls.getMainPlanes())
	#ls.showNotPlanes()
	ls.clusterizeObjects()
	#ls.showObjects()
	ls.fitCylinder()
	ls.showFeatures()


	
	# all_objects = inlier_cloud_list[0:-1]
	# all_objects.extend(cluster_array)
	# print(all_objects)
	# o3d.visualization.draw_geometries(all_objects)



