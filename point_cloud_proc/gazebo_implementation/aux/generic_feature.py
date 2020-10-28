import open3d as o3d
import numpy as np
import random
import copy 
from aux.cylinder import Cylinder
from aux.plane import Plane
from aux.cuboid import Cuboid
from aux import *

class Generic_feature:

    def __init__(self, feat, ground_normal = [0, 0, 0]):
        self.ground_normal = ground_normal
        self.feat = feat
        self.running_geo = {"total": 0, "plane":0, "cylinder":0, "cuboid":0}


    def verifyCorrespondence(self, compare_feat):
        if isinstance(self.feat,Plane):
            if isinstance(compare_feat.feat,Plane):


                normal_feature = np.asarray([self.feat.equation[0], self.feat.equation[1], self.feat.equation[2]])
                normal_candidate = np.asarray([compare_feat.feat.equation[0], compare_feat.feat.equation[1], compare_feat.feat.equation[2]])
                # Align normals
                bigger_axis = np.argmax(np.abs(normal_feature))
                if not (np.sign(normal_feature[bigger_axis]) == np.sign(normal_candidate[bigger_axis])):
                    normal_candidate = -normal_candidate
                errorNormal = (np.abs((normal_feature[0]-normal_candidate[0]))+np.abs((normal_feature[1]-normal_candidate[1]))+np.abs((normal_feature[2]-normal_candidate[2])))
                
                if(errorNormal>0.3):
                    return False
                else:

                    d = aux.distance_from_points_to_plane( compare_feat.feat.centroid, self.feat.equation)
                    print("DISTANCIA DO PLANO PRA CENTROIDE DO OUTRO PLANO: "+str(d))
                    if np.abs(d[0]) > 0.2:
                        return False
                    else:
                        d_maior = np.amax([self.feat.width,self.feat.height, compare_feat.feat.width,compare_feat.feat.height])
                        if(np.linalg.norm((self.feat.centroid - compare_feat.feat.centroid)) < d_maior):
                            #print("Encontrou correspondencia")
                            #print("Original feature: "+str(self.feat.equation))
                            #print("Candidate feature: "+str(compare_feat.feat.equation))
                            #print("Erro "+str(errorNormal))
                            self.feat.append_plane(compare_feat.feat.points_main)
                            self.running_geo["plane"] = self.running_geo["plane"]+1
                            self.running_geo["total"] = self.running_geo["total"]+1
                            return True
            if isinstance(compare_feat.feat,Cylinder):
                cyl = compare_feat.feat
                pla = self.feat
                plane_height_cylinder_normal = pla.get_height(cyl.normal)
                cylinder_height = cyl.height[1]-cyl.height[0]
                print("Altura plano")
                print(plane_height_cylinder_normal)
                print("Altura cilindro")
                print(cylinder_height)
                if(np.abs(plane_height_cylinder_normal - cylinder_height) > 0.5):
                    return False
                else:
                    print("Verificando distância entre plano e cilindro")
                    centroid_plane_to_cylinder_axis = aux.distance_from_points_to_axis(pla.centroid, cyl.normal, cyl.center)
                    print(centroid_plane_to_cylinder_axis)
                    if((np.abs(centroid_plane_to_cylinder_axis[0]) > cyl.radius*1.2) ):
                        return False
                    else:
                        print("Encontrou correspondencia")
                        self.running_geo["cylinder"] = self.running_geo["cylinder"]+1
                        self.running_geo["total"] = self.running_geo["total"]+1
                        return True
                

        if isinstance(self.feat,Cylinder):
            if isinstance(compare_feat.feat,Cylinder):
                if(np.linalg.norm(self.feat.center - compare_feat.feat.center) > 1):
                    return False
                else:
                    if(self.feat.radius - compare_feat.feat.radius)>0.5:
                        return False
                    else:
                        #print("Encontrou correspondencia")
                        #print("Original feature: "+str(self.feat.center))
                        #print("Candidate feature: "+str(compare_feat.feat.center))
                        #print("Original feature radius: "+str(self.feat.radius))
                        #print("Candidate feature radius: "+str(compare_feat.feat.radius))
                        self.running_geo["cylinder"] = self.running_geo["cylinder"]+1
                        self.running_geo["total"] = self.running_geo["total"]+1
                        return True
            else:
                pla= compare_feat.feat
                cyl= self.feat
                plane_height_cylinder_normal = pla.get_height(cyl.normal)
                cylinder_height = cyl.height[1]-cyl.height[0]
                print("Altura plano")
                print(plane_height_cylinder_normal)
                print("Altura cilindro")
                print(cylinder_height)
                if(np.abs(plane_height_cylinder_normal - cylinder_height) > 0.5):
                    return False
                else:
                    print("Verificando distância entre plano e cilindro")
                    centroid_plane_to_cylinder_axis = aux.distance_from_points_to_axis(pla.centroid, cyl.normal, cyl.center)
                    print(centroid_plane_to_cylinder_axis)
                    if((np.abs(centroid_plane_to_cylinder_axis[0]) > cyl.radius*1.2) ):
                        return False
                    else:
                        print("Encontrou correspondencia")
                        self.running_geo["cylinder"] = self.running_geo["cylinder"]+1
                        self.running_geo["total"] = self.running_geo["total"]+1
                        return True

        if isinstance(self.feat,Cuboid):
            if isinstance(compare_feat.feat,Plane):
                dim_cuboid = np.asarray([self.feat.width, self.feat.depth, self.feat.height])
                dimax = np.amax(dim_cuboid)
                # Verifica se o centroide do plano está em um raio de alcance de maior dimensão do centroide do cuboide
                if(np.linalg.norm(self.feat.centroid - compare_feat.feat.centroid) < dimax*1.2):
                    compara1 = dim_cuboid - compare_feat.feat.width
                    compara2 = dim_cuboid - compare_feat.feat.height
                    dimproxima1 = np.amin(np.abs(compara1))
                    dimproxima2 = np.amin(np.abs(compara2))
                    if(dimproxima1 < 0.1 or dimproxima2 < 0.1):
                        print("Encontrou correspondencia NO CUBOIDE")
                        self.running_geo["cuboid"] = self.running_geo["cuboid"]+1
                        self.running_geo["total"] = self.running_geo["total"]+1
                        return True




    def getProprieties(self):
        prop = self.feat.getProrieties()
        prop["runner_geo"] = self.running_geo
        return prop