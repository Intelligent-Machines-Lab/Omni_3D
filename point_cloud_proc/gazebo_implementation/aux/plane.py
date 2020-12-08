import open3d as o3d
import numpy as np
import random
import copy 
from aux import *
from aux.qhull_2d import *
from aux.min_bounding_rect import *
import matplotlib.pyplot as plt
class Plane:

    def __init__(self):
        self.inliers = []
        self.inliersId = []
        self.equation = []
        self.color = []
        self.nPoints = 0
        self.centroid = []


    def findPlane(self, pts, thresh=0.05, minPoints=100, maxIteration=1000):
        n_points = pts.shape[0]
        self.nPoints = n_points
        #print(n_points)
        best_eq = []
        best_inliers = []
        valid = False

        for it in range(maxIteration):
            # Samples 3 random points 
            id_samples = random.sample(range(1, n_points-1), 3)
            #print(id_samples)
            pt_samples = pts[id_samples]
            #print(pt_samples)

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1,:] - pt_samples[0,:]
            vecB = pt_samples[2,:] - pt_samples[0,:]

            #print(vecA)
            #print(vecB)

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)
            

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1,:]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]
            
            #print(plane_eq)

            # Distance from a point to a plane 
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            pt_id_inliers = [] # list of inliers ids
            dist_pt = (plane_eq[0]*pts[:,0]+plane_eq[1]*pts[:, 1]+plane_eq[2]*pts[:, 2]+plane_eq[3])/np.sqrt(plane_eq[0]**2+plane_eq[1]**2+plane_eq[2]**2)
            
            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if(len(pt_id_inliers) > len(best_inliers)):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
        self.inliers = pts[best_inliers]
        self.inliersId = best_inliers
        self.equation = best_eq
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
            pcd = pcd.voxel_down_sample(voxel_size=0.1)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1)
            pcd = pcd.select_by_index(ind)
            #aux.display_inlier_outlier(pcd, ind)
            #aux.display_inlier_outlier(pcd, ind)
            self.inliers = np.asarray(pcd.points)
            #self.inliersId = ind
            self.equation = best_eq
            self.centroid = np.mean(self.inliers, axis=0)

        if(self.equation):
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
            else:
                #print("PLANO INVÁLIDO")
                valid = False

        return best_eq, best_inliers, valid

    def move(self, rotMatrix=[[1,0,0],[0, 1, 0],[0, 0, 1]], tranlation=[0, 0, 0]):
        self.inliers = np.dot(self.inliers, rotMatrix.T) + tranlation
        self.points_main = np.dot(self.points_main, rotMatrix.T) + tranlation
        vec = np.dot(rotMatrix, [self.equation[0], self.equation[1], self.equation[2]]) #+ tranlation
        self.centroid = np.mean(self.inliers, axis=0)
        #d = self.equation[3] + np.dot(vec, tranlation)
        d = -np.sum(np.multiply(vec, self.centroid))
        self.equation = [vec[0], vec[1], vec[2], d]
        center_point, rot_angle, width, height, inliers_plano_desrotacionado = self.update_geometry(self.points_main)
        self.center2d = center_point
        self.rot_angle = rot_angle
        self.width = width
        self.height = height
        self.points_main = inliers_plano_desrotacionado
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
        return mesh_box


    def append_plane(self, plano, nvezes):
        #print("Shape antes de append: "+str(self.inliers.shape[0]))
        
        # #print("Shape depois de append: "+str(self.inliers.shape[0]))
        # centroid_pontos = np.mean(points, axis=0)
        # center_point, rot_angle, width, height, inliers_plano_desrotacionado = self.update_geometry(points)
        # centroid_retangulo = np.mean(inliers_plano_desrotacionado, axis=0)


        # dimin = np.amin([width, height])
        # if(np.linalg.norm(centroid_pontos-centroid_retangulo)<dimin*0.1):
        usa_media = True
        points = plano.feat.points_main
        if(usa_media):
            eqplano2 = plano.feat.equation
            nvezes_plano2 = plano.running_geo["total"]
            eqplano1 = self.equation

            # Deixa normal do plano no mesmo sentido:
            if not (np.sign(eqplano2[3]) == np.sign(eqplano1[3])):
                eqplano2 = -np.asarray(eqplano2)


            # nova equação do plano:
            # Média ponderada entre o o número de vezes já detectado e da área de cada plano
            # print('eqplano1: ', eqplano1, ' nvezes: ', nvezes+1)
            # print('eqplano2: ', eqplano2, 'nvezes_plano2: ', nvezes_plano2)

            area1 = self.width*self.height
            area2 = plano.feat.width*plano.feat.height

            self.equation = (np.asarray(eqplano1)*nvezes*area1 + np.asarray(eqplano2)*nvezes_plano2*area2)/((nvezes*area1+nvezes_plano2*area2))
            print("JUNTANDO AS EQUAÇÃO TUDO: ",self.equation)

            # Muda os dois planos para essa orientação e posição:
            #self.points_main = aux.rodrigues_rot(self.points_main, [eqplano1[0], eqplano1[1], eqplano1[2]], [self.equation[0], self.equation[1], self.equation[2]])
            #points = aux.rodrigues_rot(points, [eqplano2[0], eqplano2[1], eqplano2[2]], [self.equation[0], self.equation[1], self.equation[2]])

        provisorio = copy.deepcopy(np.append(self.points_main, points, axis=0))
        center_point, rot_angle, width, height, inliers_plano_desrotacionado = self.update_geometry(provisorio)
        self.center2d = center_point
        self.rot_angle = rot_angle
        self.width = width
        self.height = height
        self.points_main = inliers_plano_desrotacionado
        self.centroid = np.mean(self.points_main, axis=0)
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