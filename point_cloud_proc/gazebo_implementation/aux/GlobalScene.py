import open3d as o3d
import numpy as np
import random
import copy 
import time
import matplotlib.pyplot as plt
from aux.LocalScene import LocalScene
from aux.cylinder import Cylinder
from aux.plane import Plane
from aux.generic_feature import Generic_feature
from aux.aux import *
import _thread

class GlobalScene:

    def __init__(self):
        self.list_scenes = []
        self.scenes_rotation = [] # List of euclidian angles
        self.scenes_translation = [] # list of translations

        self.pcd_total = []
        self.groundNormal = []

        self.features_objects = []
        self.fet_geo = []

        self.ground_normal = []


    def add_pcd(self, pcd, commands_odom_linear, commands_odom_angular, duration):
        last_loc = np.asarray([0, 0, 0])
        last_angulo = np.asarray([0, 0, 0])
        if not len(self.scenes_translation) == 0:
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
        #print("atual_loc: "+ str(atual_loc))
        #print("atual_angulo: "+ str(atual_angulo))

        from_camera_to_inertial(pcd)

        # print("AAAAAAAAAAAAAAAAAAAh")
        # print(get_rotation_matrix_bti(atual_angulo).T)
        ls = LocalScene(pcd)
        self.ground_normal = ls.groundNormal
        #Ransac total
        ls.findMainPlanes()
        ls.defineGroundNormal()
        o3d.visualization.draw_geometries(ls.getMainPlanes())
        #ls.showNotPlanes()
        ls.clusterizeObjects()
        #ls.showObjects()
        ls.fitCylinder()
        ls.findSecundaryPlanes()
        ls.custom_draw_geometry()


        scene_features=[]
        
        for x in range(len(ls.mainPlanes)):
            ls.mainPlanes[x].move(get_rotation_matrix_bti(atual_angulo), atual_loc)
            gfeature = Generic_feature(ls.mainPlanes[x])
            scene_features.append(gfeature)
            

        for x in range(len(ls.mainCylinders)):
            ls.mainCylinders[x].move(get_rotation_matrix_bti(atual_angulo), atual_loc)
            gfeature = Generic_feature(ls.mainCylinders[x])
            scene_features.append(gfeature)


        for x in range(len(ls.secundaryPlanes)):
            ls.secundaryPlanes[x].move(get_rotation_matrix_bti(atual_angulo), atual_loc)
            gfeature = Generic_feature(ls.secundaryPlanes[x])
            scene_features.append(gfeature)

        if(len(self.features_objects)>0):
            for i_cena in range(len(scene_features)):
                ja_existe = False
                for i_global in range(len(self.features_objects)):
                    if(self.features_objects[i_global].verifyCorrespondence(scene_features[i_cena])):
                        ja_existe = True
                        break
                if(not ja_existe):
                    self.features_objects.append(scene_features[i_cena])
                    self.fet_geo.append(scene_features[i_cena].feat.get_geometry())
        else:
            for i_cena in range(len(scene_features)):
                self.features_objects.append(scene_features[i_cena])
                self.fet_geo.append(scene_features[i_cena].feat.get_geometry())


        self.scenes_rotation.append(atual_angulo)
        self.scenes_translation.append(atual_loc)
        self.fet_geo.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]).rotate(get_rotation_matrix_bti(atual_angulo), center=(0,0,0)).translate(atual_loc))

        o3d.visualization.draw_geometries(self.fet_geo, mesh_show_back_face=True)



    def getProprieties(self):
        vet_mainPlanes = []
        vet_mainCylinders = []
        for x in range(len(self.features_objects)):
            if isinstance(self.features_objects[x].feat,Plane):
                vet_mainPlanes.append(self.features_objects[x].feat.getProrieties())
            else:
                vet_mainCylinders.append(self.features_objects[x].feat.getProrieties())
        
        vet_secondaryCylinders = []
        return {"groundNormal": self.groundNormal, "planes": vet_mainPlanes, "cylinders": vet_mainCylinders, "secundaryplanes": vet_secondaryCylinders}
        #pcd_moved = pcd_moved.voxel_down_sample(voxel_size=0.1)
        
        # self.pcd_total.append(mesh_frame)

        # o3d.visualization.draw_geometries(self.pcd_total)











