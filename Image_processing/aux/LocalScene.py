import open3d as o3d
import numpy as np
import random
import copy 
import matplotlib.pyplot as plt
from aux.plane import Plane
from aux.aux import *

class LocalScene:

    def __init__(self, pointCloud):
        self.pointCloud = pointCloud # Total point cloud
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
            best_eq, best_inliers = p.findPlane(points, thresh=0.05, minPoints=100, maxIteration=1000)
            qtn_inliers = best_inliers.shape[0]
            if(qtn_inliers < 40000):
                break
            self.mainPlanes.append(p)
            outlier_cloud = outlier_cloud.select_by_index(best_inliers, invert=True)


    def showMainPlanes(self):
        pointCloudList = []
        for i in range(len(self.mainPlanes)):
            print(i)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.mainPlanes[i].inliers)
            print(self.mainPlanes[i].equation)
            pointCloudList.append(pcd.paint_uniform_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
        if(self.groundNormal != []):
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.5)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.mainPlanes[self.groundID].inliers)
            centerPCD = pcd.get_center()
            print(centerPCD)
            mesh.rotate(get_rotationMatrix_from_vectors([0, 0, 1], self.groundNormal), center=(0, 0, 0)).translate(centerPCD)
            pointCloudList.append(mesh)
        o3d.visualization.draw_geometries(pointCloudList)

    def defineGroundNormal(self):
        normalCandidatesY = []
        for i in range(len(self.mainPlanes)):
            normalCandidatesY.append(abs(self.mainPlanes[i].equation[1]))
        valMax = max(normalCandidatesY)
        idMax = normalCandidatesY.index(valMax)
        if(valMax > 0.85):
            self.groundNormal = np.asarray([self.mainPlanes[idMax].equation[0], self.mainPlanes[idMax].equation[1], self.mainPlanes[idMax].equation[2]])
            self.groundID = idMax
        print("Ground normal: "+str(self.groundNormal))