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


	def find(self, pts, thresh=0.2, minPoints=50, maxIteration=5000, useRANSAC = True, forceAxisVector = []):
		
		n_points = pts.shape[0]
		if useRANSAC:
			print(n_points)
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
		else:
			centroid = np.median(pts, axis=0)
			centroid[0] = np.min(pts[:,0])+(np.max(pts[:,0])-np.min(pts[:,0]))/2
			centroid[1] = np.min(pts[:,1])+ (np.max(pts[:,1])-np.min(pts[:,1]))/2
			centroid[2] = np.min(pts[:,2])+(np.max(pts[:,2])-np.min(pts[:,2]))/2
			print(centroid)
			vecC_stakado =  np.stack([forceAxisVector]*n_points,0)
			dist_pt = np.cross(vecC_stakado, (centroid- pts))
			dist_pt = np.linalg.norm(dist_pt, axis=1)
			radius_mean = np.mean(dist_pt)
			radius_std = np.std(dist_pt)
			radius = radius_mean+2*radius_std
			self.center = centroid
			self.normal = forceAxisVector
			self.radius = radius
			self.inliers = pts

		# Calculate heigh from center
		pts_Z = rodrigues_rot(pts, self.normal, [0,0,1])
		center_Z = rodrigues_rot(self.center, self.normal, [0,0,1])[0]
		centered_pts_Z = pts_Z[:, 2] - center_Z[2]

		self.height = [np.min(centered_pts_Z), np.max(centered_pts_Z)]


		return self.center, self.normal, self.radius,  self.inliers, self.height 


	def getProrieties(self):
		return {"center": self.center, "axis": self.normal,"radius": self.radius,"height": self.height, "color": self.color}