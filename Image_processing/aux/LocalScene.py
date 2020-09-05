import open3d as o3d
import numpy as np
import random
import copy 
import matplotlib.pyplot as plt
from aux.plane import Plane
from aux.cylinder import Cylinder
from aux.aux import *

class LocalScene:

    def __init__(self, pointCloud):
        self.pointCloud = pointCloud # Total point cloud
        self.pointCloud_notMainPlanes = []
        self.pointCloud_objects = []

        self.mainPlanes = [] # List of Plane
        self.mainCylinders = [] # List of Cylinder

        self.groundNormal = [] # Vector normal to the ground
        self.groundID = 0

    def findMainPlanes(self):
        outlier_cloud = copy.deepcopy(self.pointCloud)
        inlier_cloud_list = []
        while(True):
            # Ransac planar
            points = np.asarray(outlier_cloud.points)
            p = Plane()
            best_eq, best_inliers = p.findPlane(points, thresh=0.06, minPoints=100, maxIteration=1000)
            qtn_inliers = best_inliers.shape[0]
            if(qtn_inliers < 20000):
                break
            self.mainPlanes.append(p)

            outlier_cloud = outlier_cloud.select_by_index(best_inliers, invert=True)
            cl, ind = outlier_cloud.remove_radius_outlier(nb_points=500, radius=0.12)
            #display_inlier_outlier(outlier_cloud, ind)
            outlier_cloud = outlier_cloud.select_by_index(ind)
        self.pointCloud_notMainPlanes = outlier_cloud


    def getMainPlanes(self):
        pointCloudList = []
        for i in range(len(self.mainPlanes)):
            #print(i)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.mainPlanes[i].inliers)
            #print(self.mainPlanes[i].equation)
            pointCloudList.append(pcd.paint_uniform_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
        if(self.groundNormal != []):
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.5)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.mainPlanes[self.groundID].inliers)
            centerPCD = pcd.get_center()
            #print(centerPCD)
            mesh.rotate(get_rotationMatrix_from_vectors([0, 0, 1], self.groundNormal), center=(0, 0, 0)).translate(centerPCD)
            pointCloudList.append(mesh)
        return pointCloudList
        #o3d.visualization.draw_geometries(pointCloudList)

    def showNotPlanes(self):
        o3d.visualization.draw_geometries([self.pointCloud_notMainPlanes])

    def showObjects(self):
        o3d.visualization.draw_geometries(self.pointCloud_objects)



    def defineGroundNormal(self):
        normalCandidatesY = []
        for i in range(len(self.mainPlanes)):
            normalCandidatesY.append(abs(self.mainPlanes[i].equation[1]))
        valMax = max(normalCandidatesY)
        idMax = normalCandidatesY.index(valMax)
        if(valMax > 0.85):
            self.groundNormal = np.asarray([self.mainPlanes[idMax].equation[0], self.mainPlanes[idMax].equation[1], self.mainPlanes[idMax].equation[2]])
            if(self.groundNormal[1] < 0):
                self.groundNormal = -self.groundNormal
            self.groundID = idMax
        print("Ground normal: "+str(self.groundNormal))


    def clusterizeObjects(self):
        filtered_not_planes = self.pointCloud_notMainPlanes
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(filtered_not_planes.cluster_dbscan(eps=0.07, min_points=200, print_progress=False))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")

        o3d.visualization.draw_geometries([filtered_not_planes])
        cluster_array = []
        for n_cluster in range(max_label+1):
            index_from_cluster = np.where(labels == n_cluster)[0]
            cluster = filtered_not_planes.select_by_index( index_from_cluster.tolist())
            cluster_qnt_points = np.asarray(cluster.points).shape[0]
            if(cluster_qnt_points > 2000):
                self.pointCloud_objects.append(cluster)


    def fitCylinder(self):
        for i_obj in range(len(self.pointCloud_objects)):
            cyl = Cylinder()
            points = np.asarray(self.pointCloud_objects[i_obj].points)
            cyl.find(points, thresh=0.05, maxIteration=1000, forceAxisVector = self.groundNormal, useRANSAC = False)
            self.mainCylinders.append(cyl)

    def getCylinders(self, maxRadius= 9999999, showPointCloud = True):
        cymesh = []
        for i_obj in range(len(self.mainCylinders)):
            if(self.mainCylinders[i_obj].radius < maxRadius):
                R = get_rotationMatrix_from_vectors([0, 0, 1], self.mainCylinders[i_obj].normal)
                mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=self.mainCylinders[i_obj].radius, height=(self.mainCylinders[i_obj].height[1]-self.mainCylinders[i_obj].height[0]))
                mesh_cylinder.compute_vertex_normals()
                mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
                mesh_cylinder = mesh_cylinder.rotate(R, center=[0, 0, 0])
                mesh_cylinder = mesh_cylinder.translate((self.mainCylinders[i_obj].center[0], self.mainCylinders[i_obj].center[1], self.mainCylinders[i_obj].center[2]))
                cymesh.append(mesh_cylinder)
        obcylinder = []     
        if(showPointCloud):
            obcylinder = copy.deepcopy(self.pointCloud_objects)
            obcylinder.extend(cymesh)
        else:
            obcylinder = cymesh
        return obcylinder
        #o3d.visualization.draw_geometries(obcylinder)


    def showFeatures(self):
        feat = self.getMainPlanes()
        feat.extend(self.getCylinders(showPointCloud=False))
        o3d.visualization.draw_geometries(feat)