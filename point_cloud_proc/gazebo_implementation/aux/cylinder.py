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

        self.store_point_bucket = True

        self.bucket = o3d.geometry.PointCloud()
        self.bucket.points = o3d.utility.Vector3dVector([])

        self.bucket_pos = o3d.geometry.PointCloud()
        self.bucket_pos.points = o3d.utility.Vector3dVector([])

        self.bucket_odom = o3d.geometry.PointCloud()
        self.bucket_odom.points = o3d.utility.Vector3dVector([])


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

    def move(self, ekf):

        ekf = copy.deepcopy(ekf)
        atual_loc = [ekf.x_m[0,0], ekf.x_m[1,0], 0]
        atual_angulo = [0, 0, ekf.x_m[2,0]]
        rotMatrix = get_rotation_matrix_bti(atual_angulo)
        translation = atual_loc

        #print("Centro antes: "+str(self.center))
        self.center = np.dot(rotMatrix, self.center) + translation
        #print("Centro depois: "+str(self.center))
        self.inliers = np.dot(self.inliers, rotMatrix.T) + translation
        self.normal = np.dot(rotMatrix, self.normal)

        if self.store_point_bucket:
            self.bucket.points = o3d.utility.Vector3dVector(np.asarray(self.inliers))
            self.bucket_pos.points = o3d.utility.Vector3dVector(np.asarray(self.inliers))

            ekf_odom_x = copy.deepcopy(ekf.x_errado)
            atual_loc_odom = [ekf_odom_x[0,0], ekf_odom_x[1,0], 0]
            atual_angulo_odom = [0, 0, ekf_odom_x[2,0]]
            rotMatrix_odom = get_rotation_matrix_bti(atual_angulo_odom)
            tranlation_odom = atual_loc_odom
            inlier_move_odom = np.dot(self.inliers, rotMatrix_odom.T) + tranlation_odom
            self.bucket_odom.points = o3d.utility.Vector3dVector(np.asarray(inlier_move_odom))

    def append_cylinder(self, compare_feat, Z_new):
        #print("Centro antes: "+str(self.center))
        diff = np.asarray(self.center) - np.asarray([Z_new[0,0], Z_new[1,0], Z_new[2,0]])
        self.center = [Z_new[0,0], Z_new[1,0], Z_new[2,0]]
        #print("Centro depois: "+str(self.center))

        if self.store_point_bucket:
            self.bucket_pos.points = o3d.utility.Vector3dVector(np.append(self.bucket_pos.points, compare_feat.feat.inliers, axis=0))

            diff_feat_observada = np.asarray(compare_feat.feat.center) - np.asarray([Z_new[0,0], Z_new[1,0], Z_new[2,0]])
            inliers_observado_corrected = compare_feat.feat.inliers - diff_feat_observada
            inliers_feature_corrected = self.bucket.points - diff
            corrected_points =np.append(inliers_feature_corrected, inliers_observado_corrected, axis=0)
            self.bucket.points = o3d.utility.Vector3dVector(corrected_points)

            self.bucket_odom.points = o3d.utility.Vector3dVector(np.append(self.bucket_odom.points, compare_feat.feat.bucket_odom.points, axis=0))

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


    def get_high_level_feature(self):
        self.get_best_cuboid()

    def get_best_cuboid(self):
        outlier_cloud = copy.deepcopy(self.bucket)
        outlier_cloud.points = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.points) - np.asarray(self.center))
        nponto = np.asarray(outlier_cloud.points).shape[0]
        o3d.visualization.draw_geometries([outlier_cloud])
        inlier_cloud_list = []
        plane_list = []
        while(True):
            
            
            plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05, ransac_n=10,num_iterations=1000)
            plane_list.append({   "model": plane_model,
                                  "inliers": copy.deepcopy(outlier_cloud).select_by_index(inliers).points
                              })
            inlier_cloud_list.append(copy.deepcopy(outlier_cloud).select_by_index(inliers).paint_uniform_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]))
            qtn_inliers = np.asarray(inliers).shape[0]
            if(qtn_inliers < int(0.05*nponto)):
                break
            out = copy.deepcopy(outlier_cloud).select_by_index(inliers, invert=True)
            points = np.asarray(out.points)
            if(points.shape[0] < int(0.004*nponto)):
                break
            outlier_cloud = out

        o3d.visualization.draw_geometries(inlier_cloud_list)

        if len(plane_list) >= 3:
            print("Has more then 3 elements")
            best_ground_paralel = {"model":[0, 0, 0, 0], "inliers":[]}


            # 1- Find best plane paralel to ground
            for plane_item in plane_list:
                print(plane_item['model'])
                if plane_item["model"][2] > 0.9:
                    if len(best_ground_paralel["inliers"]) < len(plane_item['inliers']):
                        best_ground_paralel = plane_item
            if not best_ground_paralel["inliers"]:
                print("Não encontrou plano paralelo com o chão")

            print("bestaralel to ground:", best_ground_paralel["model"])

            # 2 -Find the model from perpendicular face
            second_face = {"model":[0, 0, 0, 0], "inliers":[]}
            for plane_item in plane_list:
                if all(plane_item["model"] != best_ground_paralel["model"]):
                    normal1 = np.asarray([best_ground_paralel["model"][0],best_ground_paralel["model"][1],best_ground_paralel["model"][2]])
                    normal2 = np.asarray([plane_item["model"][0],plane_item["model"][1],plane_item["model"][2]])
                    perpendicularity = np.cross(normal1,normal2)
                    if(np.linalg.norm(perpendicularity) > 0.9):
                        second_face = plane_item
                        print("Encontrou segunda face")
                        break
            if not second_face["inliers"]:
                print("Não tem outra face 2")

            # 3 -Verify existence of a third face perpendicular to both faces
            third_face = {"model":[0, 0, 0, 0], "inliers":[]}
            for plane_item in plane_list:
                if not (all(plane_item["model"] == best_ground_paralel["model"]) or all(plane_item["model"] == second_face["model"])):
                    print("TENTOI AQUI ", plane_item["model"])
                    normal1 = np.asarray([best_ground_paralel["model"][0],best_ground_paralel["model"][1],best_ground_paralel["model"][2]])
                    normal2 = np.asarray([second_face["model"][0],second_face["model"][1],second_face["model"][2]])
                    normal3 = np.asarray([plane_item["model"][0],plane_item["model"][1],plane_item["model"][2]])
                    perpendicularity = np.cross(normal1,normal2)
                    expected_normal3 = perpendicularity/np.linalg.norm(perpendicularity)
                    if(np.abs(np.dot(expected_normal3, normal3)) > 0.9):
                        third_face = plane_item
                        print("Encontrou terceira face")
                        break
            if not third_face["inliers"]:
                print("Não tem outra face 3")

            if best_ground_paralel and second_face and third_face:
                # Fita retângulo de menor área
                print("inlierts ",best_ground_paralel["inliers"])

                # verify which plane from cuboid has more points
                # bigger_plane = best_ground_paralel
                # if len(bigger_plane["inliers"]) < len(second_face["inliers"]):
                #     bigger_plane = second_face

                # if len(bigger_plane["inliers"]) < len(third_face["inliers"]):
                #     bigger_plane = third_face

                height = (self.height[1]-self.height[0])
                print("height: ", height)
                # (rot_angle, area, width, height, center_point, corner_points) = get_plane_segment_info(best_ground_paralel["inliers"], best_ground_paralel["model"])
                # print("plano1 ",width, " - ", height , " inliers: ",len(best_ground_paralel["inliers"]))



                # Plane 2 will have two dimensions: heigh and depth7
                (p_rot_angle, p_area, p_width, p_height, p_center_point, p_corner_points) = get_plane_segment_info(second_face["inliers"], second_face["model"])
                print("plano2 ",p_width, " - ", p_height , " inliers: ",len(second_face["inliers"]))
                depth = p_height if (abs(p_width-height) < abs(p_height-height)) else p_width

                # Plane 3 will have three dimentions: heigh and width
                (p_rot_angle, p_area, p_width, p_height, p_center_point, p_corner_points) = get_plane_segment_info(third_face["inliers"], third_face["model"])
                print("plano3 ",p_width, " - ", p_height , " inliers: ",len(third_face["inliers"]))
                width = p_height if (abs(p_width-height) < abs(p_height-height)) else p_width


                print("height: ", height, " depth: ", depth, " width: ", width)



                # normal1 = np.asarray([best_ground_paralel["model"][0],best_ground_paralel["model"][1],best_ground_paralel["model"][2]])
                normal2 = np.asarray([second_face["model"][0],second_face["model"][1],second_face["model"][2]])
                normal3 = np.asarray([third_face["model"][0],third_face["model"][1],third_face["model"][2]])
                # p1_centroid = np.mean(np.asarray(best_ground_paralel["inliers"]), axis=0)
                p2_centroid = np.mean(np.asarray(second_face["inliers"]), axis=0)
                p3_centroid = np.mean(np.asarray(third_face["inliers"]), axis=0)
                print("p2_centroid: ",p2_centroid, " - p3_centroid: ", p3_centroid)
                # encontro = get_point_between_two_lines(normal1, normal2, p1_centroid, p2_centroid)
                # encontro2 = get_point_between_two_lines(normal1, normal3, p1_centroid, p3_centroid)
                encontro3 = get_point_between_two_lines(normal2, normal3, p2_centroid, p3_centroid)
                centroid = encontro3 + np.asarray(self.center)

                print("CENTER: ", self.center)
                print("centroid: ", centroid)

                #print("plano2 ", self.getSizePlane(second_face["inliers"]), " inliers: ",len(second_face["inliers"]))
                #print("plano3 ", self.getSizePlane(third_face["inliers"]), " inliers: ",len(third_face["inliers"]))



                mesh_box = o3d.geometry.TriangleMesh.create_box(width=width, height=depth, depth=height)
                mesh_box = mesh_box.translate(np.asarray([-width/2, -depth/2, -(self.height[1]-self.height[0])/2]))
                mesh_box = mesh_box.rotate(get_rotationMatrix_from_vectors([1, 0, 0],[second_face["model"][0],second_face["model"][1],second_face["model"][2]]), center=np.asarray([0, 0, 0]))
                mesh_box.compute_vertex_normals()
                mesh_box.paint_uniform_color(self.color)
                # center the box on the frame
                # move to the plane location
                #mesh_box = mesh_box.rotate(get_rotationMatrix_from_vectors([0, 0, 1], [0, 0, 1]), center=np.asarray([0, 0, 0]))
                mesh_box = mesh_box.translate((centroid[0], centroid[1], centroid[2]))

        
                o3d.visualization.draw_geometries([self.bucket, mesh_box])




    def getProrieties(self):
        return {"center": self.center, "axis": self.normal,"radius": self.radius,"height": self.height,"radius_mean": self.radius_mean,
                "radius_std": self.radius_std,"spread": self.spread,"nPoints": self.nPoints, "color": self.color, 
                "circulation_mean":self.circulation_mean, "circulation_std":self.circulation_std }



