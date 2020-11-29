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
from aux.cuboid import Cuboid
import threading
from tkinter import *
from tkinter import ttk
from tkinter import Toplevel
import pickle
from mpl_toolkits.mplot3d import Axes3D

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
        self.ground_equation = []
        self.lc_atual = []
        self.propwindow = []

        self.updated_global = threading.Event()
        self.updated_global.clear()
        self.iatual = 0




        



    def custom_draw_geometry(self):
        # The following code achieves the same effect as:
        # o3d.visualization.draw_geometries([pcd])
        rotatex = 500
        rotatey = 550
        rotatey2 = 100
        showimages = False
        save_image = True
        vis_original = o3d.visualization.Visualizer()

        vis_original.create_window(window_name='Original', width=960, height=540, left=0, top=0)
        vis_original.add_geometry(self.lc_atual.pointCloud)
        vis_original.get_view_control().rotate(0, rotatey)
        vis_original.get_view_control().rotate(rotatex, 0)
        vis_original.get_view_control().rotate(0, rotatey2)
        vis_original.poll_events()
        vis_original.update_renderer()
        if(save_image):
            vis_original.capture_screen_image("animations/original-"+str(self.iatual)+".png")


        feat = self.lc_atual.getMainPlanes()
        feat.extend(self.lc_atual.getCylinders(showPointCloud=False))
        feat.extend(self.lc_atual.getSecundaryPlanes())
        vis_feature = o3d.visualization.Visualizer()
        vis_feature.create_window(window_name='Feature', width=960, height=540, left=960, top=0)
        for x in range(len(feat)):
            vis_feature.add_geometry(feat[x])
        vis_feature.get_view_control().rotate(0, rotatey)
        vis_feature.get_view_control().rotate(rotatex, 0)
        vis_feature.get_view_control().rotate(0, rotatey2)
        vis_feature.poll_events()
        vis_feature.update_renderer()
        if(save_image):
            vis_feature.capture_screen_image("animations/feature-"+str(self.iatual)+".png")


        vis_feature_global = o3d.visualization.Visualizer()
        vis_feature_global.create_window(window_name='Feature global', width=960, height=540, left=960, top=540)
        for x in range(len(self.fet_geo)):
            vis_feature_global.add_geometry(self.fet_geo[x])
        vis_feature_global.get_view_control().rotate(0, rotatey)
        vis_feature_global.get_view_control().rotate(rotatex, 0)
        vis_feature_global.get_view_control().rotate(0, rotatey2*2)
        vis_feature_global.poll_events()
        vis_feature_global.update_renderer()
        if(save_image):
            vis_feature_global.capture_screen_image("animations/global-"+str(self.iatual)+".png")
        if(showimages):
            while True:
                vis_original.update_geometry(self.lc_atual.pointCloud)
                if not vis_original.poll_events():
                    break
                vis_original.update_renderer()
                #vis_original.capture_screen_image("animations/original-"+str(self.iatual)+".png")

                feat = self.lc_atual.getMainPlanes()
                feat.extend(self.lc_atual.getCylinders(showPointCloud=False))
                feat.extend(self.lc_atual.getSecundaryPlanes())
                for x in range(len(feat)):
                    vis_feature.update_geometry(feat[x])
                
                if not vis_feature.poll_events():
                    break
                vis_feature.update_renderer()
                #vis_feature.capture_screen_image("animations/feature-"+str(self.iatual)+".png")
                for x in range(len(self.fet_geo)):
                    vis_feature_global.update_geometry(self.fet_geo[x])
                if not vis_feature_global.poll_events():
                    break
                vis_feature_global.update_renderer()
                #vis_feature_global.capture_screen_image("animations/global-"+str(self.iatual)+".png")

        vis_feature_global.destroy_window()
        vis_original.destroy_window()
        vis_feature.destroy_window()

    # def draw_geometries_pick_points(self, geometries):
    #     vis = o3d.visualization.VisualizerWithEditing()
    #     vis.create_window()
    #     for geometry in geometries:
    #         vis.add_geometry(geometry)
    #     vis.run()
    #     vis.destroy_window()


    def add_pcd(self, pcd, commands_odom_linear, commands_odom_angular, duration, i):
        self.iatual = i
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

        # Propagação 
        atual_loc = last_loc + vel_linear_inertial*duration
        atual_angulo = last_angulo + vel_angular_inertial*duration
        # print("atual_loc: "+ str(atual_loc))
        # print("atual_angulo: "+ str(atual_angulo))
        # print("last_loc: "+ str(last_loc))
        # print("last_angulo: "+ str(last_angulo))
        # print("vel_linear_inertial: "+ str(vel_linear_inertial))
        # print("vel_angular_inertial: "+ str(vel_angular_inertial))
        # print("duration: "+ str(duration))


        from_camera_to_inertial(pcd)

        # print("AAAAAAAAAAAAAAAAAAAh")
        # print(get_rotation_matrix_bti(atual_angulo).T)
        ls = LocalScene(pcd)
        self.lc_atual = ls
        
        #Ransac total
        ls.findMainPlanes()
        ls.defineGroundNormal(ground_eq_reserva=self.ground_equation)
        #o3d.visualization.draw_geometries(ls.getMainPlanes())
        #ls.showNotPlanes()
        ls.clusterizeObjects()
        #ls.showObjects()
        ls.fitCylinder()
        ls.findSecundaryPlanes()
        self.ground_normal = ls.groundNormal
        self.ground_equation = ls.groundEquation
        print("GND EQUATION: ", self.ground_equation)


        scene_features=[]
        
        for x in range(len(ls.mainPlanes)):
            ls.mainPlanes[x].move(get_rotation_matrix_bti(atual_angulo), atual_loc)
            gfeature = Generic_feature(ls.mainPlanes[x], ground_equation=self.ground_equation)
            scene_features.append(gfeature)


        for x in range(len(ls.secundaryPlanes)):
            ls.secundaryPlanes[x].move(get_rotation_matrix_bti(atual_angulo), atual_loc)
            gfeature = Generic_feature(ls.secundaryPlanes[x], ground_equation=self.ground_equation)
            scene_features.append(gfeature)

        for x in range(len(ls.mainCylinders)):
            ls.mainCylinders[x].move(get_rotation_matrix_bti(atual_angulo), atual_loc)
            gfeature = Generic_feature(ls.mainCylinders[x], ground_equation=self.ground_equation)
            scene_features.append(gfeature)

        if(len(self.features_objects)>0):
            for i_cena in range(len(scene_features)):
                ja_existe = False
                list_to_delete = []
                for i_global in range(len(self.features_objects)):
                    if(self.features_objects[i_global].verifyCorrespondence(scene_features[i_cena])):
                        ja_existe = True
                        if isinstance(scene_features[i_cena].feat,Cylinder) and isinstance(self.features_objects[i_global].feat,Plane):
                            # Remove all planes that can be part of this cylinder
                            self.features_objects[i_global].feat.color = scene_features[i_cena].feat.color
                            list_to_delete.append(self.features_objects[i_global])
                            #self.features_objects.append(scene_features[i_cena])
                        #break
                if(not ja_existe):
                    self.features_objects.append(scene_features[i_cena])
                else:
                    if isinstance(scene_features[i_cena].feat,Cylinder) and list_to_delete:
                    # If already exists, delete all correspondences
                        self.features_objects = [x for x in self.features_objects if x not in list_to_delete]
                        self.features_objects.append(scene_features[i_cena])
                    
        else:
            for i_cena in range(len(scene_features)):
                self.features_objects.append(scene_features[i_cena])
                #self.fet_geo.append(scene_features[i_cena].feat.get_geometry())
        self.fet_geo = []
        # Map cleaning and feature merge
        # Detect cuboid
        found_cuboid = True
        while found_cuboid:
            list_to_delete = []
            found_cuboid = False
            for ob in self.features_objects:
                if isinstance(ob.feat, Plane):
                    altura1 = ob.feat.get_height(self.ground_normal)
                    if altura1 > 1.5:
                        continue
                    if altura1 < 0.1:
                        continue
                    for ob2 in self.features_objects:
                        if isinstance(ob2.feat, Plane):
                            if (np.linalg.norm(ob2.feat.centroid - ob2.feat.centroid) > 3):
                                continue

                            altura2 = ob2.feat.get_height(self.ground_normal)
                            if altura2 > 1.5:
                                continue
                            # For a cuboid, height must be similar for the box's walls
                            if(np.abs(ob2.feat.width-ob.feat.width) < 0.0001):
                                # plane1 = plane2
                                continue
                            #print("Comparando alturas: ", altura1, " - ", altura2, " - diff: ",np.abs(altura1-altura2), " - Margem: ",((altura1+altura2)/2)*0.2)

                            #if(np.abs(altura1-altura2)<((altura1+altura2)/2)*0.2):
                            normal1 = np.asarray([ob.feat.equation[0],ob.feat.equation[1],ob.feat.equation[2]])
                            normal2 = np.asarray([ob2.feat.equation[0],ob2.feat.equation[1],ob2.feat.equation[2]])
                            perpendicularity = np.cross(normal1,normal2)
                            # Planes are perpndiculars
                            if(np.linalg.norm(perpendicularity) > 0.9):
                                # Lets search for another plane with a similar normal
                                for ob3 in self.features_objects:
                                    if isinstance(ob3.feat, Plane):
                                        altura3 = ob3.feat.get_height(self.ground_normal)
                                        if altura2 > 1.5:
                                            continue
                                        if (np.linalg.norm(ob3.feat.centroid - ob2.feat.centroid) > 3):
                                            continue
                                        if (np.linalg.norm(ob3.feat.centroid - ob.feat.centroid) > 3):
                                            continue
                                        normal3 = np.asarray([ob3.feat.equation[0],ob3.feat.equation[1],ob3.feat.equation[2]])
                                        if(np.linalg.norm(perpendicularity - normal3) < 0.3):
                                            d1 = distance_from_two_lines(normal1, normal2, ob.feat.centroid, ob2.feat.centroid)
                                            d2 = distance_from_two_lines(normal1, normal3, ob.feat.centroid, ob3.feat.centroid)
                                            #print("TESTOU AQUI DENTRO - ", d1, " - ", d2)
                                            meandim = np.mean([ob.feat.width, ob.feat.height, ob2.feat.width, ob2.feat.height, ob3.feat.width, ob3.feat.height])
                                            if(np.linalg.norm(d1) < meandim and np.linalg.norm(d2) < meandim  ):
                                                cub = Cuboid(ob.feat, ob2.feat, ob3.feat, self.ground_normal)
                                                if(cub.width*cub.height*cub.depth < 1.5**3):
                                                    ob2.feat.color = ob.feat.color
                                                    ob3.feat.color = ob.feat.color
                                                    #encontro = get_point_between_two_lines(normal1, normal2, ob.feat.centroid, ob2.feat.centroid)
                                                    #encontro2 = get_point_between_two_lines(normal1, normal3, ob.feat.centroid, ob3.feat.centroid)
                                                    
                                                    g = Generic_feature(cub, self.ground_equation)
                                                    for ob_plane_clear in self.features_objects:
                                                        if isinstance(ob_plane_clear.feat, Plane):
                                                            if(g.verifyCorrespondence(ob_plane_clear)):
                                                                list_to_delete.append(ob_plane_clear)
                                                    self.features_objects.append(g)
                                                    
                                                    # list_to_delete.append(ob)
                                                    # list_to_delete.append(ob2)
                                                    # list_to_delete.append(ob3)
                                                    # TODO: Fazer correspondência entre plano e cubo
                                                    # TODO: Verificar correspondência de todas as features planares 
                                                    break
                                                else:
                                                    continue
                        if list_to_delete:
                            break
                if list_to_delete:
                    break

            # Delete merged objects
            self.features_objects = [x for x in self.features_objects if x not in list_to_delete]
            if list_to_delete:

                found_cuboid = True

        # Map cleaning
        if(i % 5 == 0):
            print("------------------------")
            print("TA FAZENDO MAP CLEANING")
            print("------------------------")
            limpou_objeto = True
            while limpou_objeto:
                list_to_delete = []
                for i_global1 in range(len(self.features_objects)):
                    #if(self.features_objects[i_global1].self.running_geo[""])
                    for i_global2 in range(len(self.features_objects)):
                        if(not (i_global1 == i_global2)):
                            if(self.features_objects[i_global1].verifyCorrespondence(self.features_objects[i_global2])):
                                list_to_delete.append(self.features_objects[i_global2])
                                break
                    if list_to_delete:
                        break
                if list_to_delete:
                    self.features_objects = [x for x in self.features_objects if x not in list_to_delete]
                    limpou_objeto = True
                else:
                    limpou_objeto = False




        self.scenes_rotation.append(atual_angulo)
        self.scenes_translation.append(atual_loc)
        #self.fet_geo = []
        for ob in self.features_objects:
            if(ob.running_geo["total"] >= 2):
                self.fet_geo.append(ob.feat.get_geometry())

        self.fet_geo.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]).rotate(get_rotation_matrix_bti(atual_angulo), center=(0,0,0)).translate(atual_loc))

        #ls.custom_draw_geometry()
        if(i >=0):#126):
            threading.Thread(target=self.custom_draw_geometry, daemon=True).start()
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #posi = np.asarray(self.scenes_translation)
            #print(posi)
            #ax.plot3D(posi[:, 0], posi[:, 1], posi[:, 2], 'gray')
            #plt.show()
            #ax.plot(atual_loc[:, 0],atual_loc[:, 1],atual_loc[:, 2])

        f = open('feat.pckl', 'wb')
        pickle.dump(self.getProprieties(), f)
        f.close()
        #threading.Thread(target=self.showGUI, args=(self.getProprieties(),), daemon=True)
        #self.custom_draw_geometry()

        
        



    def getProprieties(self):
        vet_mainPlanes = []
        vet_mainCylinders = []
        vet_mainCuboids = []
        for x in range(len(self.features_objects)):
            if isinstance(self.features_objects[x].feat,Plane):
                vet_mainPlanes.append(self.features_objects[x].getProprieties())
            elif isinstance(self.features_objects[x].feat,Cylinder):
                vet_mainCylinders.append(self.features_objects[x].getProprieties())
            else:
                vet_mainCuboids.append(self.features_objects[x].getProprieties())

        
        vet_secondaryCylinders = []
        return {"groundNormal": self.groundNormal, "planes": vet_mainPlanes, "cylinders": vet_mainCylinders, "cuboids": vet_mainCuboids}
        #pcd_moved = pcd_moved.voxel_down_sample(voxel_size=0.1)
        
        # self.pcd_total.append(mesh_frame)

        # o3d.visualization.draw_geometries(self.pcd_total)











