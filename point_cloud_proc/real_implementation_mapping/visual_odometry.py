import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import glob
import copy 
import pandas as pd
import os
from aux.aux import *





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


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size=0.05):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(True), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 1000))
    return result
nomepasta = "dataset2"
df = pd.read_csv(nomepasta+"/data.txt", index_col=False)
df.columns =[col.strip() for col in df.columns]

list_depth = sorted(glob.glob(nomepasta+"/*_depth.png"))
list_rgb = sorted(glob.glob(nomepasta+"/*_rgb.png"))

icp_method = "normal"
first_estimate_method = "odometry"

transformationList = [] # Should be n-1 images

for a in range(len(list_rgb)-1):
    i = a
    color_raw = o3d.io.read_image(nomepasta+"/"+str(i)+"_rgb.png")
    depth_raw = o3d.io.read_image(nomepasta+"/"+str(i)+"_depth.png")
    pc1 = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = True)

    color_raw = o3d.io.read_image(nomepasta+"/"+str(i+1)+"_rgb.png")
    depth_raw = o3d.io.read_image(nomepasta+"/"+str(i+1)+"_depth.png")
    pc2 = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = False)

    T = np.identity(4)
    voxel_size = 0.05 # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pc1, pc2, voxel_size)
    #if(first_estimate_method == "RANSAC"):
    # Global registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    current_transformation = result_ransac.transformation
    glob_reg = current_transformation
    rot_matrix = glob_reg[:3,:3]
    yaw = np.arctan2(rot_matrix[1][0],rot_matrix[0][0])
    pitch = np.arctan2(-rot_matrix[2][0],np.sqrt(rot_matrix[2][1]**2+rot_matrix[2][2]**2))
    roll = np.arctan2(rot_matrix[2][1],rot_matrix[2][2])
    #print("Global registration: ", glob_reg)
    print("RANSAC ------------------")
    print("yaw: ", yaw)
    print("pitch: ", pitch)
    print("roll: ", roll)
    print("X: ", glob_reg[0,3])
    print("Y: ", glob_reg[1,3])
    print("Z: ", glob_reg[2,3])
    #else:
    t_dur = df['t_command'].values[i+1]
    c_linear = [df['c_linear_x'].values[i+1],
                df['c_linear_y'].values[i+1], df['c_linear_z'].values[i+1]]
    c_angular = [df['c_angular_x'].values[i+1], df['c_angular_y'].values[i+1], df['c_angular_z'].values[i+1]]
    T[:3,:3] = get_rotation_matrix_bti([c_angular[1]*t_dur,c_angular[2]*t_dur, c_angular[0]*t_dur]).T
    T[0,3] = c_linear[2]*t_dur
    T[1,3] = c_linear[1]*t_dur
    T[2,3] = c_linear[0]*t_dur # Odometria X significa aumento do Z
    odom_transf = T
    #print("Odometry: ", odom_transf)
    rot_matrix = odom_transf[:3,:3]
    yaw = np.arctan2(rot_matrix[1][0],rot_matrix[0][0])
    pitch = np.arctan2(-rot_matrix[2][0],np.sqrt(rot_matrix[2][1]**2+rot_matrix[2][2]**2))
    roll = np.arctan2(rot_matrix[2][1],rot_matrix[2][2])
    print("Odometry ------------------")
    print("yaw: ", yaw)
    print("pitch: ", pitch)
    print("roll: ", roll)
    print("X: ", odom_transf[0,3])
    print("Y: ", odom_transf[1,3])
    print("Z: ", odom_transf[2,3])



    voxel_radius = [0.1, 0.05, 0.04, 0.02, 0.01]
    max_iter = [50000, 10000, 5000, 300, 140]
    #current_transformation = np.identity(4)

    current_transformation = odom_transf
    
    draw_registration_result(source, target, current_transformation)
    #print(current_transformation)

    for scale in range(5):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        #print([iter, radius, scale])


        if(icp_method == "colored"):
            #print("3-3. Applying colored point cloud registration")
            result_icp = o3d.registration.registration_icp(
                source_down, target_down, radius, current_transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter))
            current_transformation = result_icp.transformation
            #print(result_icp)
        else:
            #print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)

            #print("3-2. Estimate normal.")
            source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            #print("3-3. Applying normal point cloud registration")
            result_icp = o3d.registration.registration_icp(
                source_down, target_down, radius, current_transformation,
                o3d.registration.TransformationEstimationPointToPlane())
            current_transformation = result_icp.transformation
            #print(result_icp)


    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    
    rot_matrix = current_transformation[:3,:3]
    yaw = np.arctan2(rot_matrix[1][0],rot_matrix[0][0])
    pitch = np.arctan2(-rot_matrix[2][0],np.sqrt(rot_matrix[2][1]**2+rot_matrix[2][2]**2))
    roll = np.arctan2(rot_matrix[2][1],rot_matrix[2][2])
    print("ICP ------------------")
    print("yaw: ", yaw)
    print("pitch: ", pitch)
    print("roll: ", roll)
    print("X: ", current_transformation[0,3])
    print("Y: ", current_transformation[1,3])
    print("Z: ", current_transformation[2,3])
    draw_registration_result(source, target, current_transformation)

    transformationList.append(current_transformation)

arrangedList = []

for n in range(len(transformationList)):
    for m in range(n+1): # Transformações devem ser recursivas
        color_raw = o3d.io.read_image(list_rgb[m])
        depth_raw = o3d.io.read_image(list_depth[m])
        pc1 = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = False)
        pc1 = pc1.voxel_down_sample(voxel_size=0.01)
        
        if(m >= len(arrangedList)):
            arrangedList.append(pc1.transform(transformationList[n]))
            print("Adicionando vetor arrumado a Imagem "+ str(m) + " recebendo transformação " + str(n) + "")
            #print(len(arrangedList))
        else:
            arrangedList[m] = arrangedList[m].transform(transformationList[n])
            print("Aplicando transformação Imagem "+ str(m) + " recebendo transformação " + str(n) + "")
o3d.visualization.draw_geometries(arrangedList)












        