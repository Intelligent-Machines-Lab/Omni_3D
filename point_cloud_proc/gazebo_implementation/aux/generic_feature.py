import open3d as o3d
import numpy as np
import random
import copy 
from aux.cylinder import Cylinder
from aux.plane import Plane
from aux import *

class Generic_feature:

    def __init__(self, feat):
        self.feat = feat


    def verifyCorrespondence(self, compare_feat):
        if isinstance(self.feat,Plane):
            if isinstance(compare_feat.feat,Plane):

                normal_feature = np.asarray([self.feat.equation[0], self.feat.equation[1], self.feat.equation[2]])
                normal_candidate = np.asarray([compare_feat.feat.equation[0], compare_feat.feat.equation[1], compare_feat.feat.equation[2]])
                dist_feature = self.feat.equation[3]
                dist_candidate = compare_feat.feat.equation[3]
                # Align normals
                bigger_axis = np.argmax(np.abs(normal_feature))
                if (normal_feature[bigger_axis] > 0) and (normal_candidate[bigger_axis] < 0):
                    normal_candidate = -normal_candidate
                    dist_candidate = dist_candidate
                errorNormal = (np.abs((normal_feature[0]-normal_candidate[0]))+np.abs((normal_feature[1]-normal_candidate[1]))+np.abs((normal_feature[2]-normal_candidate[2])))
                
                if(errorNormal>0.3):
                    return False
                else:
                    if np.abs(dist_feature-dist_candidate) > 1:
                        return False
                    else:
                        print("Encontrou correspondencia")
                        print("Original feature: "+str(self.feat.equation))
                        print("Candidate feature: "+str(compare_feat.feat.equation))
                        print("Erro "+str(errorNormal))
                        return True
        if isinstance(self.feat,Cylinder):
            if isinstance(compare_feat.feat,Cylinder):
                if(np.linalg.norm(self.feat.center - compare_feat.feat.center) > 1):
                    return False
                else:
                    if(self.feat.radius - compare_feat.feat.radius)>0.5:
                        return False
                    else:
                        print("Encontrou correspondencia")
                        print("Original feature: "+str(self.feat.center))
                        print("Candidate feature: "+str(compare_feat.feat.center))
                        print("Original feature radius: "+str(self.feat.radius))
                        print("Candidate feature radius: "+str(compare_feat.feat.radius))
                        return True
