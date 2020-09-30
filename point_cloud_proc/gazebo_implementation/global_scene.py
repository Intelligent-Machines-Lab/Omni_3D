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
from aux.GlobalScene import GlobalScene
from aux.plane import Plane
import pandas as pd
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


list_depth = sorted(glob.glob("gazebo_dataset/*_depth.png"))
list_rgb = sorted(glob.glob("gazebo_dataset/*_rgb.png"))

df = pd.read_csv("gazebo_dataset/data.txt")
df.columns =[col.strip() for col in df.columns]


nImages = len(df.index)

transformationList = [] # Should be n-1 images
gc = GlobalScene()
last_angx = df['ang_x'].values[0]
last_angy = df['ang_y'].values[0]
last_angz = df['ang_z'].values[0]

for i in range(nImages):
	color_raw = o3d.io.read_image("gazebo_dataset/"+str(i)+"_rgb.png")
	depth_raw = o3d.io.read_image("gazebo_dataset/"+str(i)+"_depth.png")
	pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=100, showImages = False)
	o3d.visualization.draw_geometries([pcd])

	t_dur = df['t_command'].values[i]
	c_linear = [df['c_linear_x'].values[i], df['c_linear_y'].values[i], df['c_linear_z'].values[i]]
	c_angular = [df['ang_x'].values[i]-last_angx, df['ang_y'].values[i]-last_angy, df['ang_z'].values[i]-last_angz]/t_dur


	#print("angulo real: "+str(df['ang_x'].values[i]-iangx)+", "+str(df['ang_y'].values[i]-iangy)+", "+str(df['ang_z'].values[i]-iangz))



	gc.add_pcd(pcd, c_linear, c_angular, t_dur)
	last_angx = df['ang_x'].values[i]
	last_angy = df['ang_y'].values[i]
	last_angz = df['ang_z'].values[i]

	# ls = LocalScene(pcd)

	# ls.findMainPlanes()
	# ls.defineGroundNormal()
	# ls.clusterizeObjects()
	# ls.fitCylinder()
	# ls.findSecundaryPlanes()
	# _thread.start_new_thread(showGUI, (ls.getProprieties(),))
	# ls.custom_draw_geometry()





