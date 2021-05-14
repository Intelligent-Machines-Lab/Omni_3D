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
from aux.aux_ekf import *
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
        self.ground_equation = [0, 0, -1, 1.2]
        self.lc_atual = []
        self.propwindow = []

        self.updated_global = threading.Event()
        self.updated_global.clear()
        self.iatual = 0

        self.ekf = ekf()

        self.use_planes = True
        self.use_cylinders = True


    def get_feature_from_id(self, id):
        for i_global in self.features_objects:
            if id == i_global.id:
                return i_global

    def custom_draw_geometry(self):
        # The following code achieves the same effect as:
        # o3d.visualization.draw_geometries([pcd])
        rotatex = 500
        rotatey = 550
        rotatey2 = 100
        showimages = True
        save_image = False
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


    def add_pcd(self, pcd, commands_odom_linear, commands_odom_angular, duration, i, vel_pos_real=[0,0,0], vel_orienta_real=[0, 0, 0,]):
        self.iatual = i
        last_loc = np.asarray([0, 0, 0])
        last_angulo = np.asarray([0, 0, 0])
        x_m_last, P_m_last = init_x_P()
        if not len(self.scenes_translation) == 0:
            # last_loc = self.scenes_translation[-1]
            last_angulo = self.scenes_rotation[-1]
            # x_m_last = self.ekf.x_m
            # P_m_last = self.P_m


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


        vel_pos_real = np.asarray(vel_pos_real)
        vel_orienta_real = np.asarray(vel_orienta_real)
        vel_orienta_real[2] = - vel_orienta_real[2]
        vel_orienta_real[1] = - vel_orienta_real[1]
        vel_pos_real[1] = - vel_pos_real[1]
        vel_pos_real[2] = - vel_pos_real[2]

        u_real = duration*np.asarray([[vel_pos_real[0]], [vel_orienta_real[2]]])

        # print("vel_angular_body: "+str(vel_angular_body))
        # print("vel_linear_body: "+str(vel_linear_body))
        # Change to inertial frame
        # vel_angular_inertial = np.dot(get_rotation_angvel_matrix_bti(last_angulo),vel_angular_body.T)
        # vel_linear_inertial = np.dot(get_rotation_matrix_bti(last_angulo),vel_linear_body.T)


        # KALMAN FILTER --------------------------------------------

        if not (self.ekf.x_m[0,0]==0 and self.ekf.x_m[1,0]==0 and self.ekf.x_m[2,0]==0):
            mv = get_V()
            mu, sigma = 0, np.sqrt(mv[0,0]) # mean and standard deviation
            noise_x = np.random.normal(mu, sigma, 1)[0]

            mu, sigma = 0, np.sqrt(mv[1,1]) # mean and standard deviation
            noise_theta = np.random.normal(mu, sigma, 1)[0]
        else:
            print("PRIMEIRA ITERAÇÃO SEM RUÍDO")
            noise_x = 0
            noise_theta = 0

        print('noise x: \n', noise_x)
        print('noise_theta: \n', noise_theta)
        u = duration*np.asarray([[vel_linear_body.T[0] + noise_x],
                                 [vel_angular_body.T[2] + noise_theta]])

        print(u)

        self.ekf.propagate(u)
        self.ekf.update_real_odom_states(u_real, u)
        # if self.ekf.num_total_features['feature'] > 3:
        #     neweq = self.ekf.upload_plane(np.asarray([[1], [0], [0]]), 0)
        #     print('neweq: ', neweq)
            # self.ekf.upload_plane(np.asarray([[1], [0], [0]]), 1)
            # self.ekf.upload_plane(np.asarray([[1], [0], [0]]), 2)
        

        # ----------------------------------------------------------



        # print("vel_angular_inertial: "+ str(vel_angular_inertial))
        # print("vel_linear_inertial: "+ str(vel_linear_inertial))

        # Propagação 
        atual_loc = [self.ekf.x_m[0,0], self.ekf.x_m[1,0], 0]
        atual_angulo = [0, 0, self.ekf.x_m[2,0]]
        #atual_loc = last_loc + vel_linear_inertial*duration
        #atual_angulo = last_angulo + vel_angular_inertial*duration

        #print("atual_loc: "+ str(atual_loc))
        #print("atual_angulo: "+ str(atual_angulo))
        #print("last_loc: "+ str(last_loc))
        #print("last_angulo: "+ str(last_angulo))
        #print("vel_linear_inertial: "+ str(vel_linear_inertial))
        #print("vel_angular_inertial: "+ str(vel_angular_inertial))
        #print("duration: "+ str(duration))


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
        #print("GND EQUATION: ", self.ground_equation)




        

        scene_features=[]

        planes_list = []
        if self.use_planes:
            planes_list.extend(ls.mainPlanes.copy())
            planes_list.extend(ls.secundaryPlanes.copy())

        mundinho = []
        print('mundinho: ', mundinho)
        plane_list2 = copy.deepcopy(planes_list)
        for x in range(len(plane_list2)):
            plane_list2[x].move(self.ekf)
            mundinho.extend(plane_list2[x].get_geometry())
        #o3d.visualization.draw_geometries(mundinho)    


        oldstate = copy.deepcopy(self.ekf)
        for x in range(len(planes_list)):
            mundinho = []
            mundinho.extend(self.fet_geo)
            id = self.ekf.calculate_mahalanobis(planes_list[x])

            inliers_raw = planes_list[x].inliers

            #id = -1
            z_medido = np.asarray([[planes_list[x].equation[0], planes_list[x].equation[1], planes_list[x].equation[2], planes_list[x].equation[3]]]).T
            normal_feature = np.asarray([planes_list[x].equation[0], planes_list[x].equation[1], planes_list[x].equation[2]])
            bigger_axis = np.argmax(np.abs(normal_feature))
            # if bigger_axis == 2:
            #     continue

            planes_list[x].move(self.ekf)
            gfeature = Generic_feature(planes_list[x], ground_equation=self.ground_equation)
            if(not id == -1):
                older_feature = self.get_feature_from_id(id)
                if not older_feature.correspond(gfeature, self.ekf):
                    id = -1
                else:
                    measured_plane = copy.deepcopy(gfeature)
                    measured_plane.feat.color = [1, 0, 0]
                    older_feature2 = copy.deepcopy(older_feature)
                    older_feature2.feat.color = [0, 1, 0]
                    mundinho.extend(measured_plane.feat.get_geometry())
                    mundinho.extend(older_feature2.feat.get_geometry())
                    #o3d.visualization.draw_geometries(mundinho) 
                    



                # normal_feature = np.asarray([older_feature.feat.equation[0], older_feature.feat.equation[1], older_feature.feat.equation[2]])
                # normal_candidate = np.asarray([gfeature.feat.equation[0], gfeature.feat.equation[1], gfeature.feat.equation[2]])
                # # Align normals
                # bigger_axis = np.argmax(np.abs(normal_feature))
                # if not (np.sign(normal_feature[bigger_axis]) == np.sign(normal_candidate[bigger_axis])):
                #     normal_candidate = -normal_candidate
                # errorNormal = (np.abs((normal_feature[0]-normal_candidate[0]))+np.abs((normal_feature[1]-normal_candidate[1]))+np.abs((normal_feature[2]-normal_candidate[2])))
                
                # if not(errorNormal>0.3):

                # If is ground
# print("ID DO PLANO: ", self.getGroundPlaneId())
# if(id == self.getGroundPlaneId()):
#     #pass
#     older_feature.correspond(gfeature, self.ekf)
# else:

#     d_maior = np.amax([older_feature.feat.width,older_feature.feat.height, gfeature.feat.width,gfeature.feat.height])
#     if(np.linalg.norm((older_feature.feat.centroid - gfeature.feat.centroid)) < d_maior*6):
#         area1 = older_feature.feat.width*older_feature.feat.height
#         area2 = gfeature.feat.width*gfeature.feat.height
#         if (not (area1/area2 < 0.05 or area1/area2 > 20)) or id == 0:
#             if not older_feature.correspond(gfeature, self.ekf):
#                 id = -1
#         else:
#             id = -1
#     else:
#         id = -1
# # else:
# #     id = -1
            if id == -1:
                i = self.ekf.add_plane(z_medido)
                gfeature.id = i
                self.features_objects.append(gfeature)


        if self.use_cylinders:
            for x in range(len(ls.mainCylinders)):
                #i = self.ekf.add_plane(z_medido)
                #gfeature.id = i
                #self.features_objects.append(gfeature)

                cent = np.asarray([[ls.mainCylinders[x].center[0]],[ls.mainCylinders[x].center[1]],[ls.mainCylinders[x].center[2]]])
                id = self.ekf.calculate_mahalanobis(ls.mainCylinders[x])
                ls.mainCylinders[x].move(self.ekf)
                gfeature = Generic_feature(ls.mainCylinders[x], ground_equation=self.ground_equation)
                if not id == -1:
                    older_feature = self.get_feature_from_id(id)
                    if not older_feature.correspond(gfeature, self.ekf):
                        id = -1
                if id == -1:
                    i = self.ekf.add_point(cent)
                    gfeature.id = i

                    self.features_objects.append(gfeature)







            # print("Visto do corpo: ", cent.T)
            # center_inertial = apply_g_point(self.ekf.x_m, cent)
            # ls.mainCylinders[x].move(get_rotation_matrix_bti(atual_angulo), atual_loc)
            # print("Rotacionado correto: ", ls.mainCylinders[x].center)
            # print("Rotacionado pela fc g: ", center_inertial.T)
            # zp = apply_h_point(self.ekf.x_m, center_inertial)
            # print("DesRotacionado pela fc h: ", zp.T)
            # gfeature = Generic_feature(ls.mainCylinders[x], ground_equation=self.ground_equation)
            # scene_features.append(gfeature)


        ################### Associação de dados antiga
        # if(len(self.features_objects)>0):
        #     for i_cena in range(len(scene_features)):
        #         ja_existe = False
        #         list_to_delete = []
        #         for i_global in range(len(self.features_objects)):
        #             associou = self.features_objects[i_global].verifyCorrespondence(scene_features[i_cena], self.ekf)
        #             if(associou):
        #                 ja_existe = True
        #                 if isinstance(scene_features[i_cena].feat,Cylinder) and isinstance(self.features_objects[i_global].feat,Plane):
        #                     # Remove all planes that can be part of this cylinder
        #                     self.features_objects[i_global].feat.color = scene_features[i_cena].feat.color
        #                     list_to_delete.append(self.features_objects[i_global])
        #                     #self.features_objects.append(scene_features[i_cena])
        #                 #break
        #         if(not ja_existe):
        #             if isinstance(scene_features[i_cena].feat,Plane):
        #                 z_medido = apply_h_plane(self.ekf.x_m, scene_features[i_cena].feat.equation[3]*np.asarray([[scene_features[i_cena].feat.equation[0]], [scene_features[i_cena].feat.equation[1]], [scene_features[i_cena].feat.equation[2]]]))
        #                 i = self.ekf.add_plane(z_medido)
        #                 scene_features[i_cena].id = i
        #             self.features_objects.append(scene_features[i_cena])

        #         else:
        #             if isinstance(scene_features[i_cena].feat,Cylinder) and list_to_delete:
        #             # If already exists, delete all correspondences
        #                 self.features_objects = [x for x in self.features_objects if x not in list_to_delete]
        #                 self.features_objects.append(scene_features[i_cena])
                    
        # else:
        #     for i_cena in range(len(scene_features)):
        #         if isinstance(scene_features[i_cena].feat,Plane):
        #             z_medido = apply_h_plane(self.ekf.x_m, scene_features[i_cena].feat.equation[3]*np.asarray([[scene_features[i_cena].feat.equation[0]], [scene_features[i_cena].feat.equation[1]], [scene_features[i_cena].feat.equation[2]]]))
        #             i = self.ekf.add_plane(z_medido)
        #             scene_features[i_cena].id = i
        #         self.features_objects.append(scene_features[i_cena])
        #         self.fet_geo.append(scene_features[i_cena].feat.get_geometry())
        self.fet_geo = []

        # Map cleaning and feature merge
        # Detect cuboid
        # found_cuboid = True
        # while found_cuboid:
        #     list_to_delete = []
        #     found_cuboid = False
        #     for ob in self.features_objects:
        #         if isinstance(ob.feat, Plane):
        #             altura1 = ob.feat.get_height(self.ground_normal)
        #             if altura1 > 1.5:
        #                 continue
        #             if altura1 < 0.1:
        #                 continue
        #             for ob2 in self.features_objects:
        #                 if isinstance(ob2.feat, Plane):
        #                     if (np.linalg.norm(ob2.feat.centroid - ob2.feat.centroid) > 3):
        #                         continue

        #                     altura2 = ob2.feat.get_height(self.ground_normal)
        #                     if altura2 > 1.5:
        #                         continue
        #                     # For a cuboid, height must be similar for the box's walls
        #                     if(np.abs(ob2.feat.width-ob.feat.width) < 0.0001):
        #                         # plane1 = plane2
        #                         continue
        #                     #print("Comparando alturas: ", altura1, " - ", altura2, " - diff: ",np.abs(altura1-altura2), " - Margem: ",((altura1+altura2)/2)*0.2)

        #                     #if(np.abs(altura1-altura2)<((altura1+altura2)/2)*0.2):
        #                     normal1 = np.asarray([ob.feat.equation[0],ob.feat.equation[1],ob.feat.equation[2]])
        #                     normal2 = np.asarray([ob2.feat.equation[0],ob2.feat.equation[1],ob2.feat.equation[2]])
        #                     perpendicularity = np.cross(normal1,normal2)
        #                     # Planes are perpndiculars
        #                     if(np.linalg.norm(perpendicularity) > 0.9):
        #                         # Lets search for another plane with a similar normal
        #                         for ob3 in self.features_objects:
        #                             if isinstance(ob3.feat, Plane):
        #                                 altura3 = ob3.feat.get_height(self.ground_normal)
        #                                 if altura2 > 1.5:
        #                                     continue
        #                                 if (np.linalg.norm(ob3.feat.centroid - ob2.feat.centroid) > 3):
        #                                     continue
        #                                 if (np.linalg.norm(ob3.feat.centroid - ob.feat.centroid) > 3):
        #                                     continue
        #                                 normal3 = np.asarray([ob3.feat.equation[0],ob3.feat.equation[1],ob3.feat.equation[2]])
        #                                 if(np.linalg.norm(perpendicularity - normal3) < 0.3):
        #                                     d1 = distance_from_two_lines(normal1, normal2, ob.feat.centroid, ob2.feat.centroid)
        #                                     d2 = distance_from_two_lines(normal1, normal3, ob.feat.centroid, ob3.feat.centroid)
        #                                     #print("TESTOU AQUI DENTRO - ", d1, " - ", d2)
        #                                     meandim = np.mean([ob.feat.width, ob.feat.height, ob2.feat.width, ob2.feat.height, ob3.feat.width, ob3.feat.height])
        #                                     if(np.linalg.norm(d1) < meandim and np.linalg.norm(d2) < meandim  ):
        #                                         cub = Cuboid(ob.feat, ob2.feat, ob3.feat, self.ground_normal)
        #                                         if(cub.width*cub.height*cub.depth < 1.5**3):
        #                                             ob2.feat.color = ob.feat.color
        #                                             ob3.feat.color = ob.feat.color
        #                                             #encontro = get_point_between_two_lines(normal1, normal2, ob.feat.centroid, ob2.feat.centroid)
        #                                             #encontro2 = get_point_between_two_lines(normal1, normal3, ob.feat.centroid, ob3.feat.centroid)
                                                    
        #                                             g = Generic_feature(cub, self.ground_equation)
        #                                             for ob_plane_clear in self.features_objects:
        #                                                 if isinstance(ob_plane_clear.feat, Plane):
        #                                                     associou = g.verifyCorrespondence(ob_plane_clear, self.ekf)
        #                                                     if(associou):
        #                                                         list_to_delete.append(ob_plane_clear)
        #                                             self.features_objects.append(g)
                                                    
        #                                             # list_to_delete.append(ob)
        #                                             # list_to_delete.append(ob2)
        #                                             # list_to_delete.append(ob3)
        #                                             # TODO: Fazer correspondência entre plano e cubo
        #                                             # TODO: Verificar correspondência de todas as features planares 
        #                                             break
        #                                         else:
        #                                             continue
        #                 if list_to_delete:
        #                     break
        #         if list_to_delete:
        #             break

        #     # Delete merged objects
        #     self.features_objects = [x for x in self.features_objects if x not in list_to_delete]
        #     if list_to_delete:

        #         found_cuboid = True

        #Map cleaning
        # if(i % 1 == 0):
        #     print("------------------------")
        #     print("TA FAZENDO MAP CLEANING")
        #     print("------------------------")
        #     limpou_objeto = True
        #     while limpou_objeto:
        #         list_to_delete = []
        #         for i_global1 in range(len(self.features_objects)):
        #             #if(self.features_objects[i_global1].self.running_geo[""])
        #             for i_global2 in range(len(self.features_objects)):
        #                 if(not (i_global1 == i_global2)):
        #                     associou = self.features_objects[i_global1].verifyCorrespondence(self.features_objects[i_global2], self.ekf)
        #                     if(associou):
        #                         self.ekf.delete_feature(self.features_objects[i_global2].id)
        #                         # Move todos os índices maiores que aquele pra frente
        #                         for ifeat in self.features_objects:
        #                             if self.features_objects[i_global2].id < ifeat.id:
        #                                 ifeat.id = ifeat.id-1
        #                         list_to_delete.append(self.features_objects[i_global2])
        #                         break
        #             if list_to_delete:
        #                 break
        #         if list_to_delete:
        #             self.features_objects = [x for x in self.features_objects if x not in list_to_delete]
        #             limpou_objeto = True
        #         else:
        #             limpou_objeto = False



        # pcd_moved = copy.deepcopy(pcd).rotate(get_rotation_matrix_bti(atual_angulo), center=(0,0,0)).translate(atual_loc)
        # self.pcd_total.append(pcd_moved)
        # o3d.visualization.draw_geometries(self.pcd_total)

        self.scenes_rotation.append(atual_angulo)
        self.scenes_translation.append(atual_loc)
        #self.fet_geo = []
        for ob in self.features_objects:
            if(ob.running_geo["total"] >= 0):
                print("EQUACAO FINAL: ", ob.feat.equation)
                self.fet_geo.extend(ob.feat.get_geometry())

        self.fet_geo.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]).rotate(get_rotation_matrix_bti(atual_angulo), center=(0,0,0)).translate(atual_loc))

        #ls.custom_draw_geometry()
        if(i >=100):#126):
            threading.Thread(target=self.custom_draw_geometry, daemon=True).start()
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #posi = np.asarray(self.scenes_translation)
            #print(posi)
            #ax.plot3D(posi[:, 0], posi[:, 1], posi[:, 2], 'gray')
            #plt.show()
            #ax.plot(atual_loc[:, 0],atual_loc[:, 1],atual_loc[:, 2])
        

        self.ekf.save_file()
        

        self.showPoints()
        f = open('feat.pckl', 'wb')
        pickle.dump(self.getProprieties(), f)
        f.close()
        #threading.Thread(target=self.showGUI, args=(self.getProprieties(),), daemon=True)
        #self.custom_draw_geometry()

        
        
    def getGroundPlaneId(self):
        modelGround = np.asarray([[0], [0], [1.2]])
        for id in range(self.ekf.num_total_features['feature']):
            if(self.ekf.type_feature_list[id] == self.ekf.types_feat['plane']):
                # The first feature that are most similar with ground model is considered ground
                #print("Dist planos ground: ", np.linalg.norm(np.absolute(self.ekf.get_feature_from_id(id)) - np.abs(modelGround)))
                #print("Plano analizado: ", np.absolute(self.ekf.get_feature_from_id(id)))
                if np.linalg.norm(np.absolute(self.ekf.get_feature_from_id(id)) - np.abs(modelGround)) < 0.8:
                    return id



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





    def showPoints(self):
        global_bucket = []
        for x in range(len(self.features_objects)):
            if isinstance(self.features_objects[x].feat,Plane):
                self.features_objects[x].feat.bucket_odom.paint_uniform_color(self.features_objects[x].feat.color)
                global_bucket.append(self.features_objects[x].feat.bucket_odom)
            elif isinstance(self.features_objects[x].feat,Cylinder):
                self.features_objects[x].feat.bucket_odom.paint_uniform_color(self.features_objects[x].feat.color)
                global_bucket.append(self.features_objects[x].feat.bucket_odom)
        o3d.visualization.draw_geometries(global_bucket)

        global_bucket = []
        for x in range(len(self.features_objects)):
            if isinstance(self.features_objects[x].feat,Plane):
                self.features_objects[x].feat.bucket_pos.paint_uniform_color(self.features_objects[x].feat.color)
                global_bucket.append(self.features_objects[x].feat.bucket_pos)
            elif isinstance(self.features_objects[x].feat,Cylinder):
                self.features_objects[x].feat.bucket_pos.paint_uniform_color(self.features_objects[x].feat.color)
                global_bucket.append(self.features_objects[x].feat.bucket_pos)
        o3d.visualization.draw_geometries(global_bucket)

        global_bucket = []
        for x in range(len(self.features_objects)):
            if isinstance(self.features_objects[x].feat,Plane):
                self.features_objects[x].feat.bucket.paint_uniform_color(self.features_objects[x].feat.color)
                global_bucket.append(self.features_objects[x].feat.bucket)
            elif isinstance(self.features_objects[x].feat,Cylinder):
                self.features_objects[x].feat.bucket.paint_uniform_color(self.features_objects[x].feat.color)
                global_bucket.append(self.features_objects[x].feat.bucket)

        o3d.visualization.draw_geometries(global_bucket)


        for x in range(len(self.features_objects)):
            if isinstance(self.features_objects[x].feat,Cylinder):
                self.features_objects[x].feat.get_high_level_feature()


        # global_bucket = []
        # for x in range(len(self.features_objects)):
        #     if isinstance(self.features_objects[x].feat,Plane):
        #         self.features_objects[x].feat.bucket.paint_uniform_color(self.features_objects[x].feat.color)
        #         pontos = copy.deepcopy(self.features_objects[x].feat.bucket)
        #         pontos.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=10))
        #         radii = [0.1, 0.3]
        #         rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pontos, o3d.utility.DoubleVector(radii))
        #         global_bucket.append(rec_mesh)
        #         print("passou aqui")

        # o3d.visualization.draw_geometries(global_bucket, mesh_show_back_face=True)
        





