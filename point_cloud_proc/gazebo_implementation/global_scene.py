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
import pandas as pd
import threading #thread module imported
import traceback
import _thread

nomepasta = "gazebo_dataset3"
list_depth = sorted(glob.glob(nomepasta+"/*_depth.png"))
list_rgb = sorted(glob.glob(nomepasta+"/*_rgb.png"))

df = pd.read_csv(nomepasta+"/data.txt")
df.columns =[col.strip() for col in df.columns]


nImages = len(df.index)

transformationList = [] # Should be n-1 images
gc = GlobalScene()
last_angx = df['ang_x'].values[0]
last_angy = df['ang_y'].values[0]
last_angz = df['ang_z'].values[0]

#vis_feature = o3d.visualization.Visualizer()
#vis_feature.create_window(width=960, height=540, left=960, top=0)

#threading.Thread(target=gc.update_feature_geometry, args=(vis_feature,), daemon=True).start()

# t = threading.Thread(target=showGUI, daemon=True)
# t.start()


for a in range(nImages):
    i = a+0
    color_raw = o3d.io.read_image(nomepasta+"/"+str(i)+"_rgb.png")
    depth_raw = o3d.io.read_image(nomepasta+"/"+str(i)+"_depth.png")
    pcd = open_pointCloud_from_rgb_and_depth(
        color_raw, depth_raw, meters_trunc=100, showImages = False)
    #o3d.visualization.draw_geometries([pcd])

    t_dur = df['t_command'].values[i]
    c_linear = [df['c_linear_x'].values[i],
                df['c_linear_y'].values[i], df['c_linear_z'].values[i]]

    diff_angx = np.arctan2(np.sin(df['ang_x'].values[i]-last_angx), np.cos(df['ang_x'].values[i]-last_angx))
    diff_angy = np.arctan2(np.sin(df['ang_y'].values[i]-last_angy), np.cos(df['ang_y'].values[i]-last_angy))
    diff_angz = np.arctan2(np.sin(df['ang_z'].values[i]-last_angz), np.cos(df['ang_z'].values[i]-last_angz))
    c_angular = [diff_angx, diff_angy, diff_angz]/t_dur

    #print("angulo real: "+str(df['ang_x'].values[i]-iangx)+", "+str(df['ang_y'].values[i]-iangy)+", "+str(df['ang_z'].values[i]-iangz))
    print("----------")
    print("Foto ", i)
    print("----------")
    gc.add_pcd(pcd, c_linear, c_angular, t_dur, i)

    last_angx = df['ang_x'].values[i]
    last_angy = df['ang_y'].values[i]
    last_angz = df['ang_z'].values[i]

    # ls = LocalScene(pcd)

    # ls.findMainPlanes()
    # ls.defineGroundNormal()
    # ls.clusterizeObjects()
    # ls.fitCylinder()
    # ls.findSecundaryPlanes()

    # ls.custom_draw_geometry()





