import open3d as o3d
import numpy as np
import random
import copy 
from aux import *
import aux.aux_ekf as a_ekf
from aux.qhull_2d import *
from aux.min_bounding_rect import *
import matplotlib.pyplot as plt
import pickle



class Plane:

    def __init__(self):
        self.inliers = []
        self.inliersId = []
        self.equation = []
        self.color = []
        self.nPoints = 0
        self.centroid = []


    def findPlane(self, pts, thresh=0.05, minPoints=3, maxIteration=1000):
        n_points = pts.shape[0]
        self.nPoints = n_points
        #print(n_points)
        best_eq = []
        best_inliers = []
        valid = False

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        plane_model, inliers = pcd.segment_plane(distance_threshold=thresh,ransac_n=3,num_iterations=maxIteration)
        [a, b, c, d] = plane_model
        best_eq = [a, b, c, d]
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        self.inliers = np.asarray(pts[inliers])
        self.inliersId = np.asarray(inliers)
        self.equation = [a, b, c, d]
        self.centroid = np.mean(self.inliers, axis=0)

        #print("Plano tem esse número de pontos como inliers: ", self.inliers.shape[0])
        if(int(self.inliers.shape[0]) > 2000):

        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(self.inliers)
        #     with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        #         labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=int(self.inliers.shape[0]/400), print_progress=False))

        #     max_label = labels.max()
        #     colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        #     colors[labels < 0] = 0
        #     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        #     o3d.visualization.draw_geometries([pcd])
        #     if(max_label > 1):
        #         self.equation = []
        #         self.best_inliers = []


            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.inliers)
            #o3d.visualization.draw_geometries([pcd])
            pcd = pcd.voxel_down_sample(voxel_size=0.1)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1)
            pcd = pcd.select_by_index(ind)
            #o3d.visualization.draw_geometries([pcd])
            #aux.display_inlier_outlier(pcd, ind)
            #aux.display_inlier_outlier(pcd, ind)
            self.inliers = np.asarray(pcd.points)
            #self.inliersId = ind
            self.equation = best_eq
            self.centroid = np.mean(self.inliers, axis=0)

        if(self.equation):
            print("Antes de negativar: ", self.equation)
            if self.equation[3] < 0:
                print("Negativou")
                self.equation[0] = -self.equation[0]
                self.equation[1] = -self.equation[1]
                self.equation[2] = -self.equation[2]
                self.equation[3] = -self.equation[3]
            print("Depois de negativar: ", self.equation)
            # # Simplificação em plano xy ou plano z
            # print("eq: ", self.equation)
            # vec_eq = [self.equation[0], self.equation[1], self.equation[2]]
            # imin = vec_eq.index(min(vec_eq))
            # vec_eq[imin] = 0
            # vec_eq = vec_eq / np.linalg.norm(vec_eq)
            # self.equation[0], self.equation[1], self.equation[2] = vec_eq[0], vec_eq[1], vec_eq[2]
            # print("nova eeq: ", self.equation)

            centroid_pontos = np.mean(self.inliers, axis=0)
            center_point, rot_angle, width, height, inliers_plano_desrotacionado = self.update_geometry(self.inliers)
            centroid_retangulo = np.mean(inliers_plano_desrotacionado, axis=0)
            dimin = np.amin([width, height])
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(self.inliers)
            # mesh_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]).translate(centroid_pontos)
            # mesh_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]).translate(centroid_retangulo)
            # o3d.visualization.draw_geometries([mesh_frame1, mesh_frame2, pcd])
            if(np.linalg.norm(centroid_pontos-centroid_retangulo)<dimin*0.3):
                #print("PLANO VÁLIDO")

                self.center2d = center_point
                self.rot_angle = rot_angle
                self.width = width
                self.height = height
                self.points_main = inliers_plano_desrotacionado
                self.centroid = np.mean(self.points_main, axis=0)
                valid = True

                # GATE DE VALIDAÇÃO DE ÁREA
                if self.width * self.height < 2:
                    valid = False
                    
            else:
                #print("PLANO INVÁLIDO")
                valid = False

        if valid:
            print("Saiu do plano: ", self.equation)
        return self.equation, self.inliersId, valid

    def move(self, ekf):
        ekf = copy.deepcopy(ekf)
        atual_loc = [ekf.x_m[0,0], ekf.x_m[1,0], 0]
        atual_angulo = [0, 0, ekf.x_m[2,0]]
        rotMatrix = aux.get_rotation_matrix_bti(atual_angulo)
        tranlation = atual_loc


        # inlin = np.dot(self.inliers, rotMatrix.T) + tranlation
        # pmain = np.dot(self.points_main, rotMatrix.T) + tranlation
        # cent = np.mean(inlin, axis=0)
        # vec = np.dot(rotMatrix, [self.equation[0], self.equation[1], self.equation[2]])
        # d = -np.sum(np.multiply(vec, cent))
        # eqcerta = [vec[0], vec[1],vec[2], d]
        # print("EQUAÇÃO CERTAAAAAAA:   ", eqcerta)
        # uv = d*np.asarray([[vec[0]], [vec[1]],[vec[2]]])


        for point in self.points_main:
            print("USANDO G: ",a_ekf.apply_g_point(ekf.x_m, np.asarray([point]).T).T)



        self.inliers = np.dot(self.inliers, rotMatrix.T) + tranlation

        
        self.points_main = np.dot(self.points_main, rotMatrix.T)
        #print('points_main antes: ', self.points_main)
        self.points_main = self.points_main + tranlation
        print('points_main depois: ', self.points_main)
        
        self.centroid = np.mean(self.inliers, axis=0)

        Z = np.asarray([[self.equation[0]],[self.equation[1]],[self.equation[2]], [self.equation[3]]])

        N = a_ekf.apply_g_plane(ekf.x_m, Z)
        # Z2 = a_ekf.apply_h_plane(ekf.x_m, N)

        # N2 = a_ekf.apply_g_plane(ekf.x_m, Z2)
        # Z3 = a_ekf.apply_h_plane(ekf.x_m, N2)

        # print("Z1: ", Z.T)
        # print("Z2: ", Z2.T)
        # print("Z3: ", Z3.T)


        print("USANDO GGGGGGGG: ", N.T)

        self.equation = [N[0,0], N[1,0], N[2,0], N[3,0]]#[eqcerta[0],eqcerta[1],eqcerta[2],eqcerta[3]] # #
        # if self.equation[3] < 0:
        #     self.equation[0] = self.equation[0]*-1
        #     self.equation[1] = self.equation[1]*-1
        #     self.equation[2] = self.equation[2]*-1
        #     self.equation[3] = self.equation[3]*-1
        print("EQUAÇÃO USAAAAAADAAAAA:   ", self.equation)

        center_point, rot_angle, width, height, inliers_plano_desrotacionado = self.update_geometry(self.points_main)
        self.center2d = center_point
        self.rot_angle = rot_angle
        self.width = width
        self.height = height
        #self.points_main = inliers_plano_desrotacionado
        self.centroid = np.mean(self.points_main, axis=0)


    def getProrieties(self):
        return {"equation": self.equation,"nPoints":self.inliers.shape[0], "color": self.color, "centroid":self.centroid,
                "height": self.height, "width": self.width, "center2d": self.center2d, "rot_angle":self.rot_angle}

    def get_height(self, ground_normal):
        pts_Z = aux.rodrigues_rot(self.points_main, ground_normal, [0,0,1])
        center_Z = aux.rodrigues_rot(self.points_main[4], ground_normal, [0,0,1])[0]
        centered_pts_Z = pts_Z[:, 2] - center_Z[2]
        height = np.max(centered_pts_Z) - np.min(centered_pts_Z)
        return height


    def get_geometry(self):
        center_point = np.asarray([self.center2d[0], self.center2d[1], 0])
        dep = 0.1
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=self.width, height=self.height, depth=dep)
        mesh_box = mesh_box.translate(np.asarray([-self.width/2, -self.height/2, -dep/2]))
        mesh_box = mesh_box.rotate(aux.get_rotation_matrix_bti([0, 0, self.rot_angle]), center=np.asarray([0, 0, 0]))
        mesh_box.compute_vertex_normals()
        mesh_box.paint_uniform_color(self.color)
        # center the box on the frame
        # move to the plane location
        mesh_box = mesh_box.translate(np.asarray(center_point))
        mesh_box = mesh_box.translate(np.asarray([0, 0, -self.equation[3]]))
        

        #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        #o3d.visualization.draw_geometries([mesh_frame, mesh_box])
        mesh_box = mesh_box.rotate(aux.get_rotationMatrix_from_vectors([0, 0, 1], [self.equation[0], self.equation[1], self.equation[2]]), center=np.asarray([0, 0, 0]))

        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(inliers_plano_desrotacionado)
        # pcd.voxel_down_sample(voxel_size=0.1)
        #pcd.paint_uniform_color(self.color)
        #obb = pcd.get_oriented_bounding_box()
        #obb.color = (self.color[0], self.color[1], self.color[2])
        # estimate radius for rolling ball
        #o3d.visualization.draw_geometries([pcd, mesh_box])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points_main)
        return [mesh_box, pcd]


    def append_plane(self, plano, neweq = [], nvezes=0):
        #print("Shape antes de append: "+str(self.inliers.shape[0]))
        
        # #print("Shape depois de append: "+str(self.inliers.shape[0]))
        # centroid_pontos = np.mean(points, axis=0)
        # center_point, rot_angle, width, height, inliers_plano_desrotacionado = self.update_geometry(points)
        # centroid_retangulo = np.mean(inliers_plano_desrotacionado, axis=0)


        # dimin = np.amin([width, height])
        # if(np.linalg.norm(centroid_pontos-centroid_retangulo)<dimin*0.1):
        plano = copy.deepcopy(plano)
        neweq = copy.deepcopy(neweq)
        usa_media = False
        points = plano.feat.points_main
        if(usa_media):
            eqplano2 = plano.feat.equation
            nvezes_plano2 = plano.running_geo["total"]
            eqplano1 = copy.deepcopy(self.equation)


            # nova equação do plano:
            # Média ponderada entre o o número de vezes já detectado e da área de cada plano
            # print('eqplano1: ', eqplano1, ' nvezes: ', nvezes+1)
            # print('eqplano2: ', eqplano2, 'nvezes_plano2: ', nvezes_plano2)

            area1 = self.width*self.height
            area2 = plano.feat.width*plano.feat.height

            self.equation = (np.asarray(eqplano1)*nvezes*area1 + np.asarray(eqplano2)*nvezes_plano2*area2)/((nvezes*area1+nvezes_plano2*area2))
            #print("JUNTANDO AS EQUAÇÃO TUDO: ",self.equation)

            # Muda os dois planos para essa orientação e posição:
            #self.points_main = aux.rodrigues_rot(self.points_main, [eqplano1[0], eqplano1[1], eqplano1[2]], [self.equation[0], self.equation[1], self.equation[2]])
            #points = aux.rodrigues_rot(points, [eqplano2[0], eqplano2[1], eqplano2[2]], [self.equation[0], self.equation[1], self.equation[2]])
        else:
            self.equation = neweq
        provisorio = copy.deepcopy(np.append(self.points_main, points, axis=0))
        center_point, rot_angle, width, height, inliers_plano_desrotacionado = self.update_geometry(provisorio)
        self.center2d = center_point
        self.rot_angle = rot_angle
        self.width = width
        self.height = height
        self.points_main = inliers_plano_desrotacionado
        centroidantes = self.centroid
        self.centroid = np.mean(self.points_main, axis=0)
        centroiddepois = self.centroid

        #print("DIFERENÇA DE CENTROIDES: ", np.linalg.norm(centroidantes-centroiddepois))
        discentnormal = np.dot((centroidantes-centroiddepois),np.asarray([self.equation[0], self.equation[1], self.equation[2]]))
        # O que me interessa mesmo aqui é mudança da centroide mas em direção a normal do plano. Não tem problema a centroide mudar na direção da superfície do plano
        #print("DIFERENÇA DE CENTROIDES na direção do plano: ",discentnormal)
        if(np.abs(discentnormal) > 0.8):
            self.color = (1, 0, 0)
            return False
        return True
        # else:
        #     return False


        
    def update_geometry(self, points):
        # Encontra parâmetros do semi-plano
        inlier_planez = points

        # Encontra representação 2d da projeção na normal do plano
        inliers_plano = aux.rodrigues_rot(copy.deepcopy(inlier_planez), [self.equation[0], self.equation[1], self.equation[2]], [0, 0, 1])- np.asarray([0, 0, -self.equation[3]])
        dd_plano = np.delete(inliers_plano, 2, 1)

        # Fita retângulo de menor área
        print('dd_plano: ',dd_plano.shape)

        filename = 'pontos.pckl'
        outfile = open(filename,'wb')
        pickle.dump(dd_plano,outfile)
        outfile.close()


        hull_points = qhull2D(dd_plano)
        hull_points = hull_points[::-1]
        (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)

        # Volta pro espaço 3D
        p = np.vstack((np.asarray(corner_points), np.asarray(center_point)))
        ddd_plano= np.c_[ p, np.zeros(p.shape[0]) ] + np.asarray([0, 0, -self.equation[3]])
        inliers_plano_desrotacionado = aux.rodrigues_rot(ddd_plano, [0, 0, 1], [self.equation[0], self.equation[1], self.equation[2]])
        return center_point, rot_angle, width, height, inliers_plano_desrotacionado








# # Load saved point cloud and visualize it
# pcd_load = o3d.io.read_point_cloud("caixa.ply")
# #o3d.visualization.draw_geometries([pcd_load])
# points = np.asarray(pcd_load.points)

# plano1 = Plane()

# best_eq, best_inliers = plano1.findPlane(points, 0.01)
# plane = pcd_load.select_by_index(best_inliers).paint_uniform_color([1, 0, 0])
# obb = plane.get_oriented_bounding_box()
# obb2 = plane.get_axis_aligned_bounding_box()
# obb.color = [0, 0, 1]
# obb2.color = [0, 1, 0]
# not_plane = pcd_load.select_by_index(best_inliers, invert=True)
# #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])

# o3d.visualization.draw_geometries([not_plane, plane, obb, obb2])