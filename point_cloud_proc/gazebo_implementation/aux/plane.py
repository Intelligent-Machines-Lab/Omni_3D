import open3d as o3d
import numpy as np
import random
import copy 
from aux import *
from aux.qhull_2d import *
from aux.min_bounding_rect import *

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
        print(n_points)
        best_eq = []
        best_inliers = []

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

        if(self.equation):
            self.update_geometry(self.inliers)

        return best_eq, best_inliers

    def move(self, rotMatrix=[[1,0,0],[0, 1, 0],[0, 0, 1]], tranlation=[0, 0, 0]):
        self.inliers = np.dot(self.inliers, rotMatrix.T) + tranlation
        self.points_main = np.dot(self.points_main, rotMatrix.T) + tranlation
        vec = np.dot(rotMatrix, [self.equation[0], self.equation[1], self.equation[2]]) #+ tranlation
        self.centroid = np.mean(self.inliers, axis=0)
        #d = self.equation[3] + np.dot(vec, tranlation)
        d = -np.sum(np.multiply(vec, self.centroid))
        self.equation = [vec[0], vec[1], vec[2], d]
        self.update_geometry(self.points_main)




    def getProrieties(self):
        return {"equation": self.equation,"nPoints":self.inliers.shape[0], "color": self.color, "centroid":self.centroid,
                "height": self.height, "width": self.width}

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


    def append_plane(self, points):
        #print("Shape antes de append: "+str(self.inliers.shape[0]))
        self.points_main = np.append(self.points_main, points, axis=0)
        #print("Shape depois de append: "+str(self.inliers.shape[0]))
        self.update_geometry(self.points_main)
        self.centroid = np.mean(self.points_main, axis=0)


        
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
        self.center2d = center_point
        self.rot_angle = rot_angle
        self.width = width
        self.height = height
        self.points_main = inliers_plano_desrotacionado







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