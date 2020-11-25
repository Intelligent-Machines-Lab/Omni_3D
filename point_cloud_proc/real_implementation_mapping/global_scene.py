import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import glob
import copy 
import os
import time
from tkinter import *
from tkinter import ttk
from aux.aux import *
from aux.LocalScene import LocalScene
from aux.GlobalScene import GlobalScene
from aux.plane import Plane
from aux.odometer import Odometer
import pandas as pd
import threading #thread module imported
import traceback
import _thread

pastanome = "dataset"

list_depth = sorted(glob.glob(pastanome+"/*_depth.png"))
list_rgb = sorted(glob.glob(pastanome+"/*_rgb.png"))

df = pd.read_csv(pastanome+"/data.txt", index_col=False)
df.columns =[col.strip() for col in df.columns]
#print(df)

nImages = len(df.index)

transformationList = [] # Should be n-1 images
gc = GlobalScene()
odom = Odometer(pastanome)
# last_angx = df['ang_x'].values[0]
# last_angy = df['ang_y'].values[0]
# last_angz = df['ang_z'].values[0]

#vis_feature = o3d.visualization.Visualizer()
#vis_feature.create_window(width=960, height=540, left=960, top=0)

#threading.Thread(target=gc.update_feature_geometry, args=(vis_feature,), daemon=True).start()

# t = threading.Thread(target=showGUI, daemon=True)
# t.start()


for a in range(nImages):
    i = a
    print(odom.get_odometry(i))
    color_raw = o3d.io.read_image(pastanome+"/"+str(i+1)+"_rgb.png")
    depth_raw = o3d.io.read_image(pastanome+"/"+str(i+1)+"_depth.png")
    pcd = open_pointCloud_from_rgb_and_depth(
        color_raw, depth_raw, meters_trunc=2, showImages = False)
    #o3d.visualization.draw_geometries([pcd])

    #print("angulo real: "+str(df['ang_x'].values[i]-iangx)+", "+str(df['ang_y'].values[i]-iangy)+", "+str(df['ang_z'].values[i]-iangz))
    #break
    #gc.add_pcd(pcd, c_linear, c_angular, t_dur, i)

    #last_angx = df['ang_x'].values[i]
    #last_angy = df['ang_y'].values[i]
    #last_angz = df['ang_z'].values[i]

    # ls = LocalScene(pcd)

    # ls.findMainPlanes()
    # ls.defineGroundNormal()
    # ls.clusterizeObjects()
    # ls.fitCylinder()
    # ls.findSecundaryPlanes()

    # ls.custom_draw_geometry()





