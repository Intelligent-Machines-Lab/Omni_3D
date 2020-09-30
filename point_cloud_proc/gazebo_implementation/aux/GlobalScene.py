import open3d as o3d
import numpy as np
import random
import copy 
import matplotlib.pyplot as plt
from aux.LocalScene import LocalScene
from aux.cylinder import Cylinder
from aux.aux import *

class GlobalScene:

    def __init__(self):
        self.list_scenes = []
        self.scenes_rotation = [] # List of euclidian angles
        self.scenes_translation = [] # list of translations

        self.pcd_total = []
        self.groundNormal = []

    def add_pcd(self, pcd, commands_odom_linear, commands_odom_angular, duration):
        last_loc = np.asarray([0, 0, 0])
        last_angulo = np.asarray([0, 0, 0])
        if not len(self.pcd_total) == 0:
            last_loc = self.scenes_translation[-1]
            last_angulo = self.scenes_rotation[-1]


        # print("Odometria linear: "+str(commands_odom_linear))
        # print("Odometria angular: "+str(commands_odom_angular))

        # in the body frame
        # transform commands into the right frame
        vel_angular_body = np.asarray(commands_odom_angular)
        vel_linear_body = np.asarray(commands_odom_linear)
        vel_angular_body[2] = - vel_angular_body[2]
        vel_angular_body[1] = - vel_angular_body[1]
        vel_linear_body[1] = - vel_linear_body[1]
        vel_linear_body[2] = - vel_linear_body[2]

        # print("vel_angular_body: "+str(vel_angular_body))
        # print("vel_linear_body: "+str(vel_linear_body))
        # Change to inertial frame
        vel_angular_inertial = np.dot(get_rotation_angvel_matrix_bti(last_angulo),vel_angular_body.T)
        vel_linear_inertial = np.dot(get_rotation_matrix_bti(last_angulo),vel_linear_body.T)

        # print("vel_angular_inertial: "+ str(vel_angular_inertial))
        # print("vel_linear_inertial: "+ str(vel_linear_inertial))


        atual_loc = last_loc + vel_linear_inertial*duration
        atual_angulo = last_angulo + vel_angular_inertial*duration
        print("atual_loc: "+ str(atual_loc))
        print("atual_angulo: "+ str(atual_angulo))

        # print("AAAAAAAAAAAAAAAAAAAh")
        # print(get_rotation_matrix_bti(atual_angulo).T)

        pcd = pcd.rotate(get_rotationMatrix_from_vectors([0, 0, -1], [1,0,0]), center=(0,0,0))
        pcd = pcd.rotate(get_rotationMatrix_from_vectors([0, 1, 0], [0,0,-1]), center=(0,0,0))
        #pcd.points[1] = - pcd.points[1]
        #pcd.points[2] = - pcd.points[2]
        pcd_moved = copy.deepcopy(pcd).rotate(get_rotation_matrix_bti(atual_angulo), center=(0,0,0)).translate(atual_loc)
        pcd_moved = pcd_moved.voxel_down_sample(voxel_size=0.1)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0]).rotate(get_rotation_matrix_bti(atual_angulo), center=(0,0,0)).translate(atual_loc)
        self.pcd_total.append(pcd_moved)
        self.pcd_total.append(mesh_frame)
        self.scenes_rotation.append(atual_angulo)
        self.scenes_translation.append(atual_loc)
        o3d.visualization.draw_geometries(self.pcd_total)








