import open3d as o3d
import numpy as np
import random
import copy 
from aux.aux import *

class Cylinder:

    def __init__(self):
        self.inliers = []
        self.equation = []
        self.tMatrix = [] # env to plane
        self.rMatrix = [] # env to plane
        self.color = []
        self.nPoints = 0
        self.circulation_mean = 0
        self.circulation_std = 0


    def find(self, pts, thresh=0.2, minPoints=50, maxIteration=5000, useRANSAC = True, forceAxisVector = []):
        
        n_points = pts.shape[0]
        if useRANSAC:
            best_eq = []
            best_inliers = []

            for it in range(maxIteration):
                # Samples 3 random points 
                id_samples = random.sample(range(1, n_points-1), 3)
                pt_samples = pts[id_samples]

                # We have to find the plane equation described by those 3 points
                # We find first 2 vectors that are part of this plane
                # A = pt2 - pt1
                # B = pt3 - pt1
                if (forceAxisVector == []):
                    vecA = pt_samples[1,:] - pt_samples[0,:]
                    vecA_norm = vecA / np.linalg.norm(vecA)
                    vecB = pt_samples[2,:] - pt_samples[0,:]
                    vecB_norm = vecB / np.linalg.norm(vecB)
                    #print(vecA)
                    #print(vecB)

                    # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
                    vecC = np.cross(vecA_norm, vecB_norm)
                else:
                    vecC = forceAxisVector
                vecC = vecC / np.linalg.norm(vecC)

                # Now we calculate the rotation of the points with rodrigues equation
                P_rot = rodrigues_rot(pt_samples, vecC, [0,0,1])
                #print("P_rot:")
                #print(P_rot)

                # Find center from 3 points
                # http://paulbourke.net/geometry/circlesphere/
                # Find lines that intersect the points
                # Slope:
                ma = 0
                mb = 0
                while(ma == 0):
                    ma = (P_rot[1, 1]-P_rot[0, 1])/(P_rot[1, 0]-P_rot[0, 0])
                    #print("ma: "+str(ma))
                    mb = (P_rot[2, 1]-P_rot[1, 1])/(P_rot[2, 0]-P_rot[1, 0])
                    #print("mb: "+str(mb))
                    if(ma == 0):
                        #print("ma zero, rolling order")
                        P_rot = np.roll(P_rot,-1,axis=0)
                    else:
                        break
                # Calulate the center by verifying intersection of each orthogonal line
                p_center_x = (ma*mb*(P_rot[0, 1]-P_rot[2, 1]) + mb*(P_rot[0, 0]+P_rot[1, 0]) - ma*(P_rot[1, 0]+P_rot[2, 0]))/(2*(mb-ma))
                p_center_y = -1/(ma)*(p_center_x - (P_rot[0, 0]+P_rot[1, 0])/2)+(P_rot[0, 1]+P_rot[1, 1])/2
                p_center = [p_center_x, p_center_y, 0]
                radius = np.linalg.norm(p_center - P_rot[0, :])

                # Remake rodrigues rotation
                center = rodrigues_rot(p_center, [0,0,1], vecC)[0]

                # Distance from a point to a plane 
                pt_id_inliers = [] # list of inliers ids
                vecC_stakado =  np.stack([vecC]*n_points,0)

                
                dist_pt = np.cross(vecC_stakado, (center- pts))
                dist_pt = np.linalg.norm(dist_pt, axis=1)
                #print(dist_pt)

                # Select indexes where distance is biggers than the threshold
                pt_id_inliers = np.where(np.abs(dist_pt-radius) <= thresh)[0]
                #print(len(pt_id_inliers))
                if(len(pt_id_inliers) > len(best_inliers)):
                    best_inliers = pt_id_inliers
                    self.inliers = best_inliers
                    self.center = center
                    self.normal = vecC
                    self.radius = radius
                    self.radius_mean = 0
                    self.radius_std = 0
                    self.spread = 0
        else:
            # Initial centroid and radius estimation
            centroid = np.median(pts, axis=0)
            centroid[0] = np.min(pts[:,0])+(np.max(pts[:,0])-np.min(pts[:,0]))/2
            centroid[1] = np.min(pts[:,1])+ (np.max(pts[:,1])-np.min(pts[:,1]))/2
            centroid[2] = np.min(pts[:,2])+(np.max(pts[:,2])-np.min(pts[:,2]))/2
            #print(centroid)
            vecC_stakado =  np.stack([forceAxisVector]*n_points,0)
            dist_pt = np.cross(vecC_stakado, (centroid- pts))
            dist_pt = np.linalg.norm(dist_pt, axis=1)
            radius_mean = np.mean(dist_pt)
            radius_std = np.std(dist_pt)
            radius = radius_mean+2*radius_std

            # Refine centroid and radius
            # Move centroid along the camera > centroid axis.
            # First calculate the axis https://www.maplesoft.com/support/help/Maple/view.aspx?path=MathApps/ProjectionOfVectorOntoPlane
            projNormal = np.dot((np.dot(centroid, forceAxisVector)/(np.linalg.norm(forceAxisVector)**2)), forceAxisVector)
            projPlane  = centroid - projNormal
            projPlane = projPlane / np.linalg.norm(projPlane)
            distMove = (2*radius*np.sin(70*np.pi/180)/(3*70*np.pi/180))*projPlane
            centroid = centroid + distMove

            dist_pt = np.cross(vecC_stakado, (centroid- pts))
            dist_pt = np.linalg.norm(dist_pt, axis=1)
            radius_mean = np.mean(dist_pt)
            radius_std = np.std(dist_pt)
            radius = radius_mean+2*radius_std


            self.center = centroid
            self.normal = forceAxisVector
            self.radius = radius
            self.inliers = pts
            self.radius_mean = radius_mean
            self.radius_std = radius_std
            self.spread = radius_std/radius_mean

        # Calculate heigh from center
        pts_Z = rodrigues_rot(pts, self.normal, [0,0,1])
        center_Z = rodrigues_rot(self.center, self.normal, [0,0,1])[0]
        centered_pts_Z = pts_Z[:, 2] - center_Z[2]

        self.height = [np.min(centered_pts_Z), np.max(centered_pts_Z)]
        self.nPoints = n_points

        return self.center, self.normal, self.radius,  self.inliers, self.height 


    def calculatePlanification(self, showNormal=True):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.inliers)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=500))
        pcd.orient_normals_towards_camera_location()
        pcd.normalize_normals()

        normals = np.asarray(copy.deepcopy(pcd.normals))
        vecC_stakado =  np.stack([self.normal]*self.nPoints,0)

        # Projection of normal in the plane from which the normal is the axis
        # https://stackoverflow.com/questions/35090401/how-to-calculate-the-dot-product-of-two-arrays-of-vectors-in-python
        # Same as:
        # # First calculate the axis https://www.maplesoft.com/support/help/Maple/view.aspx?path=MathApps/ProjectionOfVectorOntoPlane
        # # projNormal = np.dot((np.dot(centroid, forceAxisVector)/(np.linalg.norm(forceAxisVector)**2)), forceAxisVector)
        un = (normals*vecC_stakado).sum(1)
        m_nn_m = np.linalg.norm(self.normal)**2
        unn = (un*np.asarray(self.normal)[:,np.newaxis]).T
        projNormal = normals - np.divide(unn,m_nn_m)
        pcd.normals = o3d.utility.Vector3dVector(projNormal)
        #pcd.normalize_normals()

        pcd2 = copy.deepcopy(pcd)

        # Calculate vector perpenticular from axis to point
        dist_pt = np.cross(vecC_stakado, (self.center-self.inliers))
        dist_pt = dist_pt / np.linalg.norm(dist_pt)

        # If they are orthogonal, means they are aligned, high values are planes
        circulation = np.cross(dist_pt, projNormal)
        circulation_abs = np.linalg.norm(circulation, axis=1)
        self.circulation_mean = np.mean(circulation_abs)
        self.circulation_std = np.std(circulation_abs)

        pcd2.normals = o3d.utility.Vector3dVector(dist_pt*10)
        #pcd2.normalize_normals()
        if(showNormal):
            o3d.visualization.draw_geometries([pcd, pcd2], point_show_normal=True)

    def move(self, rotMatrix=[[1,0,0],[0, 1, 0],[0, 0, 1]], translation=[0, 0, 0]):
        #print("Centro antes: "+str(self.center))
        self.center = np.dot(rotMatrix, self.center) + translation
        #print("Centro depois: "+str(self.center))
        self.inliers = np.dot(self.inliers, rotMatrix.T) + translation
        self.normal = np.dot(rotMatrix, self.normal)

    def append_cylinder(self, compare_feat, Z_new):
        #print("Centro antes: "+str(self.center))
        diff = np.asarray(self.center) - np.asarray([Z_new[0,0], Z_new[1,0], Z_new[2,0]])
        self.center = [Z_new[0,0], Z_new[1,0], Z_new[2,0]]
        #print("Centro depois: "+str(self.center))
        self.inliers = self.inliers + diff # não sei se os pixeis tão alinhados

        self.radius = (self.radius + compare_feat.feat.radius)/2

        h_atual = (self.height[1]-self.height[0])
        h_feat = (compare_feat.feat.height[1]-compare_feat.feat.height[0])
        h_novo = (h_atual + h_feat)/2

        self.height[1] = h_novo/2
        self.height[0] = -h_novo/2

        return True

        #self.normal = np.dot(rotMatrix, self.normal)


    def get_geometry(self):
        R = get_rotationMatrix_from_vectors([0, 0, 1], self.normal)
        mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=self.radius, height=(self.height[1]-self.height[0]))
        mesh_cylinder.compute_vertex_normals()
        mesh_cylinder.paint_uniform_color(self.color)
        mesh_cylinder = mesh_cylinder.rotate(R, center=[0, 0, 0])
        #print("Centro depois2: "+str(self.center))
        mesh_cylinder = mesh_cylinder.translate((self.center[0], self.center[1], self.center[2]))
        return [mesh_cylinder]



    def getProrieties(self):
        return {"center": self.center, "axis": self.normal,"radius": self.radius,"height": self.height,"radius_mean": self.radius_mean,
                "radius_std": self.radius_std,"spread": self.spread,"nPoints": self.nPoints, "color": self.color, 
                "circulation_mean":self.circulation_mean, "circulation_std":self.circulation_std }