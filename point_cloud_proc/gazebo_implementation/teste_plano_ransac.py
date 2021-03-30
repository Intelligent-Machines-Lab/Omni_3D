import open3d as o3d
import numpy as np
import glob
from aux.aux import *

#nomepasta = "gazebo_dataset_circular_planes"
nomepasta = "gazebo_dataset_circular_cylinder1"
#nomepasta = "gazebo_dataset_simples1"
#nomepasta = "gazebo_dataset_planes4"
#nomepasta = "gazebo_dataset_planes_perpendicular"
#nomepasta = "gazebo_corredor"
#nomepasta = "gazebo_dataset_planes"
#nomepasta = "gazebo_dataset2"
list_depth = sorted(glob.glob(nomepasta+"/*_depth.png"))
list_rgb = sorted(glob.glob(nomepasta+"/*_rgb.png"))

for a in range(10):
    color_raw = o3d.io.read_image(nomepasta+"/"+str(a)+"_rgb.png")
    depth_raw = o3d.io.read_image(nomepasta+"/"+str(a)+"_depth.png")


    pcd = open_pointCloud_from_rgb_and_depth(color_raw, depth_raw, meters_trunc=5, showImages = False)

    #Ransac total
    outlier_cloud = pcd
    inlier_cloud_list = []
    planosMaximos = 1

    for x in range(planosMaximos):
        # Ransac planar
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.1,ransac_n=3,num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud_list.append(outlier_cloud.select_by_index(inliers).paint_uniform_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
        outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries(inlier_cloud_list)

    inlier_cloud_list.append(outlier_cloud)
    o3d.visualization.draw_geometries(inlier_cloud_list)