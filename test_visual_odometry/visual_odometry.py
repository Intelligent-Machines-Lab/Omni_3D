import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import glob
import copy 
import os





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
    #o3d.visualization.draw_geometries([source_temp, target_temp])
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    #o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
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
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result



list_depth = sorted(glob.glob("seq_1/*.png"), key=os.path.getmtime)
list_rgb = sorted(glob.glob("seq_1/*.jpg"), key=os.path.getmtime)

transformationList = [] # Should be n-1 images

for i in range(len(list_rgb)-1):
    color_raw = o3d.io.read_image(list_rgb[i])
    depth_raw = o3d.io.read_image(list_depth[i])
    pc1 = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = False)

    color_raw = o3d.io.read_image(list_rgb[i+1])
    depth_raw = o3d.io.read_image(list_depth[i+1])
    pc2 = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=3, showImages = False)

    voxel_size = 0.05 # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pc1, pc2, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # threshold = 0.02
    # reg_p2p = o3d.registration.registration_icp(source_down, target_down, threshold, result_ransac.transformation,
    #         o3d.registration.TransformationEstimationPointToPlane(),
    #         o3d.registration.ICPConvergenceCriteria(max_iteration = 1000000))

    # result_icp = o3d.registration.registration_icp(
 #        source_down, target_down, 0.02, result_ransac.transformation,
 #        o3d.registration.TransformationEstimationPointToPlane(), o3d.registration.ICPConvergenceCriteria(max_iteration = 50000))
    # print(result_icp)


    voxel_radius = [0.05, 0.04, 0.02]
    max_iter = [5000, 300, 140]
    #current_transformation = np.identity(4)
    current_transformation = result_ransac.transformation

    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        # print("3-1. Downsample with a voxel size %.2f" % radius)
        # source_down = source.voxel_down_sample(radius)
        # target_down = target.voxel_down_sample(radius)

        # print("3-2. Estimate normal.")
        # source_down.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        # target_down.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)

    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
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












        