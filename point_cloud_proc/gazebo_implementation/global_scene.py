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
# from aux.odometer import Odometer
import pandas as pd
import threading #thread module imported
import traceback
import _thread

#nomepasta = "gazebo_dataset_circular_planes"
#nomepasta = "gazebo_dataset_circular_cylinder1"
#nomepasta = "gazebo_dataset_simples1"
#nomepasta = "gazebo_dataset_planes4"
#nomepasta = "gazebo_dataset_planes_perpendicular"
nomepasta = "gazebo_corredor"
#nomepasta = "gazebo_dataset_planes"
#nomepasta = "gazebo_dataset2"
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

last_x = df['pos_x'].values[0]
last_y = df['pos_y'].values[0]
last_z = df['pos_z'].values[0]
# odom = Odometer(nomepasta)

use_gaussian_noise = False

first_pos = np.asarray([])
first_orienta = np.asarray([])


for a in range(nImages):
    #i = (a-1)*2+0+10
    i = a
    color_raw = o3d.io.read_image(nomepasta+"/"+str(i)+"_rgb.png")
    depth_raw = o3d.io.read_image(nomepasta+"/"+str(i)+"_depth.png")
    # dp = cv2.imread(nomepasta+"/"+str(i)+"_depth.png",cv2.IMREAD_UNCHANGED)
    # depth_raw = o3d.geometry.Image(dp.astype(np.uint16))
    #depth_raw = o3d.io.read_image(nomepasta+"/"+str(i)+"_depth.png")

    pcd = open_pointCloud_from_rgb_and_depth(
        color_raw, depth_raw, meters_trunc=10, showImages = False)

    if use_gaussian_noise:
        pontos = np.asarray(pcd.points)
        for ponto in pontos:
            z = ponto[2]
            mean = 0
            sigma = 0.001063+0.0007278*z+0.003949*z*z
            # based on Analysis  and  Noise  Modeling  of  the  Intel  RealSense  D435  for  MobileRobots
            #logistic(0,noise,
            ponto[2] = ponto[2] + np.random.normal(mean, sigma, 1)
            #ponto[2] = ponto[2] + np.random.logistic(0,sigma/10)

        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        #print("Define parameters used for hidden_point_removal")
        camera = [0, 0, diameter]
        radius = diameter *700

        #print("Get all points that are visible from given view point")
        _, pt_map = pcd.hidden_point_removal(camera, radius)

        #print("Visualize result")
        pcd = pcd.select_by_index(pt_map)
    #o3d.visualization.draw_geometries([pcd])



    # m_trans = odom.get_odometry(i-1, method="auto", icp_refine=True, max_tentativas_ransac = 10)
    # t_dur = 1
    # x,y,z,yaw,pitch,roll = get_xyz_yawpitchroll_from_transf_matrix(m_trans)
    # c_linear = np.asarray([z, x, y])
    # c_angular = np.asarray([yaw,roll,pitch])

    t_dur = 1#df['t_command'].values[i]
    c_linear = [df['c_linear_x'].values[i],
                df['c_linear_y'].values[i], df['c_linear_z'].values[i]]


    diff_x = df['pos_x'].values[i]-last_x
    diff_y = df['pos_y'].values[i]-last_y
    diff_z = df['pos_z'].values[i]-last_z
    c_linear = [np.sqrt(diff_x**2 + diff_y**2), 0, 0]#/t_dur

    diff_angx = np.arctan2(np.sin(df['ang_x'].values[i]-last_angx), np.cos(df['ang_x'].values[i]-last_angx))
    diff_angy = np.arctan2(np.sin(df['ang_y'].values[i]-last_angy), np.cos(df['ang_y'].values[i]-last_angy))
    diff_angz = np.arctan2(np.sin(df['ang_z'].values[i]-last_angz), np.cos(df['ang_z'].values[i]-last_angz))
    c_angular = [diff_angx, diff_angy, diff_angz]#/t_dur


    print("----------")
    print("Foto ", i)
    print("----------")
    gc.add_pcd(pcd, c_linear, c_angular, t_dur, i, c_linear, c_angular)

    last_x = df['pos_x'].values[i]
    last_y = df['pos_y'].values[i]
    last_z = df['pos_z'].values[i]
    last_angx = df['ang_x'].values[i]
    last_angy = df['ang_y'].values[i]
    last_angz = df['ang_z'].values[i]






