import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import glob
import copy 
import os
from tkinter import *
from tkinter import ttk
from aux.aux import *
from aux.LocalScene import LocalScene
from aux.plane import Plane
import _thread #thread module imported


def showGUI(prop):
	root = Tk()
	  
	# Using treeview widget 
	tree = ttk.Treeview(root, selectmode ='browse',height = 20) 
	# Calling pack method w.r.to treeview 
	tree.pack(fill='x')
	# Constructing vertical scrollbar 
	# with treeview 
	# verscrlbar = ttk.Scrollbar(root,  
	#                            orient ="vertical",  
	#                            command = tree.yview) 


	tree["columns"]=("value")
	tree.heading("value", text="Value")
	tree.column("value", width=900)
	color_list = []
	for key in prop: # key of proprety
		if(isinstance(prop[key], list)): # if is a list
			if((key == "planes") or (key =="cylinders") or (key =="secundaryplanes")): # we need to load more if it is plane or cylinder
				arv = tree.insert("", 1, text=key, open=True) # add major tree primitive division
				for o in range(len(prop[key])):	# Repeat for every object
					colorzinha = prop[key][o]["color"]
					color_list.append(colorzinha)
					arv2 = tree.insert(arv, 2, text=(key+" "+str(o)), tags=[str(colorzinha)]) # add plane 1, plane 2, plane 3 ....
					for key2 in prop[key][o]: # iterate over propriety
						tree.insert(arv2, "end", text=(key2), values=(str(prop[key][o][key2]),))
			else:
				arv = tree.insert("", 1, text=key, values=(str(prop[key]),))
		else:
			arv = tree.insert("", 1, text=key, values=(str(prop[key]),))

	for cor in color_list:
		mycolor = '#%02x%02x%02x' % (int(cor[0]*255), int(cor[1]*255), int(cor[2]*255))
		tree.tag_configure(str(cor), background=mycolor)
	tree.pack()
	root.mainloop()

n = 0

#list_depth = ["selecionadas_wall_box/1_depth.png"]
#list_rgb = ["selecionadas_wall_box/1_rgb.jpg"]

showNormals = True

#list_depth = sorted(glob.glob("selecionadas_wall_cylinder/*_depth.png"))
#list_rgb = sorted(glob.glob("selecionadas_wall_cylinder/*.jpg"))

list_depth = sorted(glob.glob("selecionadas_wall_box/*_depth.png"))
list_rgb = sorted(glob.glob("selecionadas_wall_box/*.jpg"))

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
	#o3d.visualization.draw_geometries(ls.getMainPlanes())
	#ls.showNotPlanes()
	ls.clusterizeObjects()
	#ls.showObjects()
	ls.fitCylinder()
	ls.findSecundaryPlanes()
	_thread.start_new_thread(showGUI, (ls.getProprieties(),))
	ls.custom_draw_geometry()

	#ls.showFeatures()
	

	
	# all_objects = inlier_cloud_list[0:-1]
	# all_objects.extend(cluster_array)
	# print(all_objects)
	# o3d.visualization.draw_geometries(all_objects)




