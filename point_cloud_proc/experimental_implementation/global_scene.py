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
import sys


pastanome = "dataset3"

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
    m_trans = odom.get_odometry(i, method="odom", icp_refine=False)
    t_dur = 1
    x,y,z,yaw,pitch,roll = get_xyz_yawpitchroll_from_transf_matrix(m_trans)
    c_linear = np.asarray([z, x, y])
    c_angular = np.asarray([yaw,roll,pitch])
    color_raw = o3d.io.read_image(pastanome+"/"+str(i+1)+"_rgb.png")
    depth_raw = o3d.io.read_image(pastanome+"/"+str(i+1)+"_depth.png")
    pcd = open_pointCloud_from_rgb_and_depth(
        color_raw, depth_raw, meters_trunc=3, showImages = False)
    #o3d.visualization.draw_geometries([pcd])

    #print("angulo real: "+str(df['ang_x'].values[i]-iangx)+", "+str(df['ang_y'].values[i]-iangy)+", "+str(df['ang_z'].values[i]-iangz))
    #break
    gc.add_pcd(pcd, c_linear, c_angular, t_dur, i+1, c_linear, c_angular)






