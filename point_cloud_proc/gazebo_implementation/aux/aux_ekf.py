import numpy as np

import pickle
from operator import itemgetter
from scipy.spatial import distance
from aux.aux import *
import aux.cylinder
#import aux.plane
import sys
# from aux.cuboid import Cuboid


class ekf:

    def __init__(self):
        self.init_angle = 0
        self.x_m, self.P_m  = init_x_P(self.init_angle)
        self.x_p, self.P_p  = init_x_P(self.init_angle)

        

        self.x_real = np.asarray([[0],[0],[self.init_angle]])
        self.x_errado = np.asarray([[0],[0],[self.init_angle]])

        self.x_p_list = []
        self.P_p_list = []

        self.x_m_list = []
        self.P_m_list = []

        self.x_real_list= []
        self.x_errado_list= []

        self.num_total_features = {'feature':0}

        self.types_feat = {'plane':1, 'point':2}
        self.type_feature_list = []


    def propagate(self, u):
        u = copy.deepcopy(u)
        x_m_last = self.x_m[:3,:]
        P_m_last = self.P_m[:3,:3]
        self.x_p = copy.deepcopy(self.x_m)
        self.P_p = copy.deepcopy(self.P_m)
        print('u ',u )
        #print("self.P_m: ", self.P_m)
        #print("x_m_last: \n",x_m_last)
        #print("P_m_last: \n",P_m_last)
        #print("u: \n",u)

        Fx = get_Fx(x_m_last, u)
        Fv = get_Fv(x_m_last, u)
        V  = get_V()


        new_x_p = apply_f(x_m_last, u) # Propagation

        #print("x_p: \n",x_p)
        new_P_p = Fx @ P_m_last @ Fx.T +  Fv @ V @ Fv.T
        #print("new_P_p: ", new_P_p)



        self.x_p[:3,:] = new_x_p

        self.P_p[:3,:3] = new_P_p
        #print("self.P_p: ", self.P_p)

        self.P_m = self.P_p
        self.x_m = self.x_p



    def add_plane(self, Z):
        i_plano = self.num_total_features['feature'] # pega id do próximo plano
        self.num_total_features['feature'] = self.num_total_features['feature']+1 # soma contador de planos
        self.type_feature_list.append(self.types_feat['plane'])
        
        print("Z: ", Z)
        Gx = get_Gx_plane(self.x_m, Z)
        Gz = get_Gz_plane(self.x_m, Z)
        Yz = get_Yz(Gx, Gz, self.P_m)
        # print('Yz ', Yz)

        N = apply_g_plane(self.x_m, Z)
        self.x_m = np.vstack((self.x_m, N))
        print("New Feature: ", N)
        print("New x: ", self.x_m)
        W = get_W_plane()
        meio_bloco = np.block([[self.P_m,                                   np.zeros((self.P_m.shape[0], W.shape[1]))],
                               [np.zeros((W.shape[0], self.P_m.shape[1])),  W]])
        # print('meio_bloco: \n', meio_bloco)

        self.P_m = Yz @ meio_bloco @ Yz.T
        return i_plano


    def add_point(self, C):
        i_plano = self.num_total_features['feature'] # pega id do próximo plano
        self.num_total_features['feature'] = self.num_total_features['feature']+1 # soma contador de planos
        self.type_feature_list.append(self.types_feat['point'])

        Gx = get_Gx_point(self.x_m, C)
        Gz = get_Gz_point(self.x_m, C)
        Yz = get_Yz(Gx, Gz, self.P_m)
        # print('Yz ', Yz)

        Z = apply_g_point(self.x_m, C)
        self.x_m = np.vstack((self.x_m, Z))
        # print("New Feature: ", N)
        # print("New x: ", self.x_m)
        W = get_W_point()
        meio_bloco = np.block([[self.P_m,                                   np.zeros((self.P_m.shape[0], W.shape[1]))],
                               [np.zeros((W.shape[0], self.P_m.shape[1])),  W]])
        # print('meio_bloco: \n', meio_bloco)

        self.P_m = Yz @ meio_bloco @ Yz.T
        return i_plano


    def upload_plane(self, Z, id, only_test=False):
        Z = normalize_plane_feature(Z)
        Z = copy.deepcopy(Z)
        # if np.argmax(np.abs(Z)) == 2:
        #     print("chão")
        #     # Z plane
        #     Z[0] = 0.0001
        #     Z[1] = 0.0001
        # else:
        #     # XY Plane
        #     Z[2] = 0.0001
        Hxv = get_Hxv_plane(self.x_m, self.x_m[(3+id*4):(3+(id+1)*4)])
        Hxp = get_Hxp_plane(self.x_m, self.x_m[(3+id*4):(3+(id+1)*4)])
        Hx = get_Hx(Hxv, Hxp, id, self.P_m)
        Hw = get_Hw_plane()


        #invz = np.asarray([[1/Z[0]], [1/Z[1]], [1/Z[2]]])
        W = get_W_plane()#*modiff

        S = Hx @ self.P_m @ Hx.T + Hw @ W @ Hw.T

        # if not np.linalg.cond(S) < 1/sys.float_info.epsilon:
        #     print("S é singular RETORNANDO VALOR SEM ATUALIZAÇÃO")
        #     print(S)
        #     return self.x_m[(3+id*3):(3+(id+1)*3)]

        K = self.P_m @ Hx.T @ np.linalg.inv(S)
        #print("Kalman gain: ", K)
        # K[0, 1] = 0 
        # K[0, 2] = 0 
        # #K[0, 3] = 0 

        # K[1, 0] = 0 
        # K[1, 2] = 0 
        # #K[1, 3] = 0 

        # K[2, 2] = 0 
        # K[2, 3] = 0 
        #print("Kalman gain 2: ", K)

        #print("Feature nova: ", Z.T)
        #print("Feature antiga: ", apply_h_plane(self.x_m, self.x_m[(3+id*4):(3+(id+1)*4)]).T)

        N = apply_h_plane(self.x_m, self.x_m[(3+id*4):(3+(id+1)*4)])
        N = normalize_plane_feature(N)

        print('SDADWAWZ[3,0] ', Z[3,0])
        print('AWDADWN[3,0] ', N[3,0])

        print("Plano no mundo antigo (visto do robo): ", (N).T)
        print("Plano medido: (visto do robo)", (Z).T)

        print("Plano no mundo antigo (visto do mundo): ", (apply_g_plane(self.x_m,N)).T)
        print("Plano medido: (visto do mundo)", (apply_g_plane(self.x_m,Z)).T)

        v =     Z - N
        print('xm[2]: ',self.x_m[2])
        # v[0,0] = np.cos(self.x_m[2])*v[0,0] + np.sqrt(1-np.cos(self.x_m[2])**2)*v[1,0]
        # v[1,0] = np.sin(self.x_m[2])*v[1,0] + np.sqrt(1-np.sin(self.x_m[2])**2)*v[0,0]
        #v[3,0] = -v[3,0]
        # print("Plano v: ", (apply_g_plane(self.x_m,Z)-apply_g_plane(self.x_m,N)).T)
        # print("Plano v: ", (v).T)

        # # TESTING FOREWARD
        # x_m_test = copy.deepcopy(self.x_m + K @ v)
        # P_m_test = copy.deepcopy(self.P_m - K @ Hx @ self.P_m)
        # x_m_test[(3+id*4):(3+(id+1)*4)][0:4] = normalize_plane_feature(x_m_test[(3+id*4):(3+(id+1)*4)][0:4])
        # novod = x_m_test[(3+id*4):(3+(id+1)*4)][3,0]
        # print("novo d v = N - Z: ", x_m_test[(3+id*4):(3+(id+1)*4)][3,0])

        # # TESTING BACKWARD
        # v = Z - N 
        # x_m_test = copy.deepcopy(self.x_m + K @ v)
        # P_m_test = copy.deepcopy(self.P_m - K @ Hx @ self.P_m)
        # x_m_test[(3+id*4):(3+(id+1)*4)][0:4] = normalize_plane_feature(x_m_test[(3+id*4):(3+(id+1)*4)][0:4])
        # novod = x_m_test[(3+id*4):(3+(id+1)*4)][3,0]
        # print("novo d v = Z - N : ", x_m_test[(3+id*4):(3+(id+1)*4)][3,0])


        # Z_outro = apply_g_plane(self.x_m,Z)
        # N_outro = apply_g_plane(self.x_m,N)
        # print('novod ', novod)
        # print('Z[3,0] ', Z_outro[3,0])
        # print('N[3,0] ', N_outro[3,0])

        # print("(novod <= MEDIDO[3,0] and novod >= ANTIGO[3,0]): ", (novod <= Z_outro[3,0] and novod >= N_outro[3,0]))
        # print("(novod >= MEDIDO[3,0] and novod <= ANTIGO[3,0]): ", (novod >= Z_outro[3,0] and novod <= N_outro[3,0]))

        # if not ((novod <= Z_outro[3,0] and novod >= N_outro[3,0]) or (novod >= Z_outro[3,0] and novod <= N_outro[3,0])):
        #     v = Z - N 
        # else:
        #     v = N - Z

        # x_m_test = copy.deepcopy(self.x_m + K @ v)
        # P_m_test = copy.deepcopy(self.P_m - K @ Hx @ self.P_m)
        # x_m_test[(3+id*4):(3+(id+1)*4)][0:4] = normalize_plane_feature(x_m_test[(3+id*4):(3+(id+1)*4)][0:4])

        # print("d escolhido : ", x_m_test[(3+id*4):(3+(id+1)*4)][3,0])

        if only_test:
            x_m_test = copy.deepcopy(self.x_m + K @ v)
            P_m_test = copy.deepcopy(self.P_m - K @ Hx @ self.P_m)
            x_m_test[(3+id*4):(3+(id+1)*4)][0:4] = normalize_plane_feature(x_m_test[(3+id*4):(3+(id+1)*4)][0:4])
            return x_m_test[(3+id*4):(3+(id+1)*4)]
        else:
            #print("Atualizando plano: ", Z.T)
            init = copy.deepcopy(self.x_m[0:3])
            print("Posição inicial: ", init.T)
            print("K @ v: ", K @ v)
            mov_kv = K @ v
            # mov_kv[0,0] = np.cos(self.x_m[2])*mov_kv[0,0] + np.sqrt(1-np.cos(self.x_m[2])**2)*mov_kv[1,0]
            # mov_kv[1,0] = np.sin(self.x_m[2])*mov_kv[1,0] + np.sqrt(1-np.sin(self.x_m[2])**2)*mov_kv[0,0]
            # mov_kv[1,0] = np.sin(self.x_m[2])*mov_kv[0,0] + np.sin(self.x_m[2])*mov_kv[1,0]

            self.x_m = self.x_m + mov_kv
            print("Posição final: ", self.x_m[0:3].T)
            print("Movimento: ",(self.x_m[0:3] - init).T)
            self.P_m = self.P_m - K @ Hx @ self.P_m

            self.x_m[(3+id*4):(3+(id+1)*4)][0:4] = normalize_plane_feature(self.x_m[(3+id*4):(3+(id+1)*4)][0:4])
            print("Plano no final: ", self.x_m[(3+id*4):(3+(id+1)*4)][0:4].T)

            # Featurenova = self.x_m[(3+id*4):(3+(id+1)*4)]
            # Featurenova_vec = Featurenova[:3,0]
            # print('Featurenova: ', Featurenova)
            # print('Featurenova vec: ', Featurenova_vec)
            # print('Modulo vec: ', np.linalg.norm(Featurenova_vec))
            # self.x_m[(3+id*4):(3+(id+1)*4)][0,0] = self.x_m[(3+id*4):(3+(id+1)*4)][0,0]/np.linalg.norm(Featurenova_vec)
            # self.x_m[(3+id*4):(3+(id+1)*4)][1,0] = self.x_m[(3+id*4):(3+(id+1)*4)][1,0]/np.linalg.norm(Featurenova_vec)
            # self.x_m[(3+id*4):(3+(id+1)*4)][2,0] = self.x_m[(3+id*4):(3+(id+1)*4)][2,0]/np.linalg.norm(Featurenova_vec)
            # self.x_m[(3+id*4):(3+(id+1)*4)][3,0] = self.x_m[(3+id*4):(3+(id+1)*4)][3,0]/np.linalg.norm(Featurenova_vec)
            # Featurenova2 = self.x_m[(3+id*4):(3+(id+1)*4)]
            # Featurenova_vec2 = Featurenova2[:3,0]
            # print('Featurenova2: ', Featurenova2)
            # print('Featurenova vec2: ', Featurenova_vec2)
            # print('Modulo vec2: ', np.linalg.norm(Featurenova_vec2))
            return copy.deepcopy(self.x_m[(3+id*4):(3+(id+1)*4)])


    def upload_point(self, Z, id, only_test=False):
        Hxv = get_Hxv_point(self.x_m, Z)
        Hxp = get_Hxp_point(self.x_m, Z)
        Hx = get_Hx(Hxv, Hxp, id, self.P_m)
        Hw = get_Hw_point()
        W = get_W_point()

        S = Hx @ self.P_m @ Hx.T + Hw @ W @ Hw.T
        K = self.P_m @ Hx.T @ np.linalg.inv(S)

        v = Z - apply_h_point(self.x_m, self.x_m[(3+id*3):(3+(id+1)*3)])

        if only_test:
            x_m_test = self.x_m + K @ v
            P_m_test = self.P_m - K @ Hx @ self.P_m
            return x_m_test[(3+id*3):(3+(id+1)*3)]
        else:
            self.x_m = self.x_m + K @ v
            self.P_m = self.P_m - K @ Hx @ self.P_m
            return self.x_m[(3+id*3):(3+(id+1)*3)]

    def calculate_mahalanobis(self, feature):
        feature = copy.deepcopy(feature)
        if isinstance(feature,aux.plane.Plane):
            eq = feature.equation
            N = np.asarray([[eq[0],eq[1],eq[2],eq[3]]]).T
            distances = []
            for id in range(self.num_total_features['feature']):
                if(self.type_feature_list[id] == self.types_feat['plane']):
                    Zp = apply_h_plane(self.x_m, self.get_feature_from_id(id))
                    Hxv = get_Hxv_plane(self.x_m, self.get_feature_from_id(id))
                    Hxp = get_Hxp_plane(self.x_m, self.get_feature_from_id(id))
                    Hx = get_Hx(Hxv, Hxp, id, self.P_m)
                    Hw = get_Hw_plane()
                    W = get_W_plane()

                    S = Hx @ self.P_m @ Hx.T + Hw @ W @ Hw.T
                    #print("ZP: ", Zp)
                    #print("N: ", N)
                    y = N - Zp
                    y = np.square(y)

                    if not np.linalg.cond(S) < 1/sys.float_info.epsilon:
                        print("S é singular Impedindo associação")
                        print(S)
                        return -1
                    d = y.T @ np.linalg.inv(S) @ y
                    d2 = distance.mahalanobis(N, Zp, np.linalg.inv(S))
                    print("PLANO: Zp: ",Zp.T, " N: ",N.T, " d: ", d[0][0], " d2: ", d2)
                    distances.append(np.sqrt(d[0][0]))
                else:
                    # If the feature is from another type, we put a very high distance
                    distances.append(99999)
            if distances:
                idmin = min(enumerate(distances), key=itemgetter(1))[0] 
                if(distances[idmin] > 15):
                    idmin = -1
            else:
                idmin = -1
        elif(isinstance(feature,aux.cylinder.Cylinder)):
            centroid = feature.center
            C = np.asarray([[centroid[0]],[centroid[1]],[centroid[2]]])
            distances = []
            for id in range(self.num_total_features['feature']):
                if(self.type_feature_list[id] == self.types_feat['point']):
                    Zp = apply_h_point(self.x_m, self.get_feature_from_id(id))
                    Hxv = get_Hxv_point(self.x_m, Zp)
                    Hxp = get_Hxp_point(self.x_m, Zp)
                    Hx = get_Hx(Hxv, Hxp, id, self.P_m)
                    Hw = get_Hw_point()
                    W = get_W_point()

                    S = Hx @ self.P_m @ Hx.T + Hw @ W @ Hw.T
                    y = C - Zp
                    y = np.square(y)
                    d = y.T @ np.linalg.inv(S) @ y
                    d2 = distance.mahalanobis(C, Zp, np.linalg.inv(S))
                    #print("CILINDRO: Zp: ",Zp.T, " N: ",C.T, " d: ", d[0][0], " d2: ", d2)
                    distances.append(np.sqrt(d[0][0]))
                else:
                    # If the feature is from another type, we put a very high distance
                    distances.append(99999)
            if distances:
                idmin = min(enumerate(distances), key=itemgetter(1))[0]
                # antes tava 16 
                if(distances[idmin] > 4):
                    idmin = -1
            else:
                idmin = -1


        # if(not id == -1):
        #     feature.move(self)
        #     gfeature = Generic_feature(feature, ground_equation=self.ground_equation)
        #     older_feature = self.get_feature_from_id(id)
        #     d_maior = np.amax([older_feature.feat.width,older_feature.feat.height, gfeature.feat.width,gfeature.feat.height])
        #     if(np.linalg.norm((older_feature.feat.centroid - gfeature.feat.centroid)) < d_maior):
        #         area1 = older_feature.feat.width*older_feature.feat.height
        #         area2 = gfeature.feat.width*gfeature.feat.height
        #         if (not (area1/area2 < 0.05 or area1/area2 > 20)) or id == 0:
        #             older_feature.correspond(gfeature, self.ekf)
        #         else:
        #             id = -1
        #     else:
        #         id = -1
        #     # else:
        #     #     id = -1


        print("Associação: ",distances, " id menor: ",idmin)
        
        return idmin


    def get_feature_from_id(self, id):

        return copy.deepcopy(self.x_m[(3+id*4):(3+(id+1)*4)])

    def delete_feature(self, id):
        # deleta linhas da matriz de estados
        self.x_m = np.delete(self.x_m,(np.s_[(3+id*3):(3+(id+1)*3)]), axis=0)

        # deleta linhas da matriz de covariância
        self.P_m = np.delete(self.P_m,(np.s_[(3+id*3):(3+(id+1)*3)]), axis=0)

        # deleta colunas da matriz de covariância
        self.P_m = np.delete(self.P_m,(np.s_[(3+id*3):(3+(id+1)*3)]), axis=1)

        self.num_total_features['plane'] = self.num_total_features['plane']-1


    def save_file(self, u_real, u):
        self.x_real = apply_f(self.x_real, u_real)
        self.x_errado = apply_f(self.x_errado, u)

        self.x_p_list.append(self.x_p.copy())
        self.P_p_list.append(self.P_p.copy())

        self.x_m_list.append(self.x_m.copy())
        self.P_m_list.append(self.P_m.copy())



        self.x_real_list.append(self.x_real.copy())
        self.x_errado_list.append(self.x_errado.copy())
        # print("f: ",self.x_p)
        # print("P: ",self.P_p)
        nsalva = {}
        nsalva['x_p_list'] =  self.x_p_list
        nsalva['P_p_list'] =  self.P_p_list

        nsalva['x_m_list'] =  self.x_m_list
        nsalva['P_m_list'] =  self.P_m_list

        nsalva['x_real_list'] =  self.x_real_list
        nsalva['x_errado_list'] =  self.x_errado_list

        f = open('ekf.pckl', 'wb')
        pickle.dump(nsalva, f)
        f.close()


def normalize_plane_feature(Z1):
    Z = copy.deepcopy(Z1)
    if Z[3,0] < 0:
        Z[0,0] = Z[0,0]*-1
        Z[1,0] = Z[1,0]*-1
        Z[2,0] = Z[2,0]*-1
        Z[3,0] = Z[3,0]*-1

    Z[0,0] = Z[0,0]/np.sqrt(np.linalg.norm(Z[0:3]))
    Z[1,0] = Z[1,0]/np.sqrt(np.linalg.norm(Z[0:3]))
    Z[2,0] = Z[2,0]/np.sqrt(np.linalg.norm(Z[0:3]))
    #Z[3,0] = Z[3,0]/np.linalg.norm(Z[0:3])

    Z= np.round(Z,4)
    return Z


def get_Hx(Hxv, Hxp, id, P_m):
    print('Hxv:\n',Hxv)
    print('Hxp:\n',Hxp)
    antes_p = np.zeros((Hxv.shape[0],id*Hxp.shape[1]))
    depois_p = np.zeros((Hxv.shape[0],(P_m.shape[1]-(Hxv.shape[1]+id*Hxp.shape[1]+Hxp.shape[1]) ) ))
    print('antes_p:\n',antes_p)
    print('depois_p:\n',depois_p)
    Hx = np.hstack((Hxv,antes_p,Hxp,depois_p))
    print('Hx: \n',Hx)
    return Hx

def get_Yz(Gx, Gz, P_m):
    n = P_m.shape[0]
    In = np.eye(n)
    # print("Gx.shape[0]", Gx.shape[1])
    # print("Gx.shape[1]", Gx.shape[0])
    zero_low = np.zeros((Gx.shape[0],n-Gx.shape[1]))
    zero_up = np.zeros((n,Gz.shape[1]))
    # print('zero_low',zero_low)
    # print('zero_up',zero_up)

    low = np.hstack((Gx,zero_low))
    low = np.hstack((low,Gz))
    up = np.hstack((In,zero_up))

    Yz = np.vstack((up,low))

    # print('Yz ', Yz)
    # print('low', low)
    # print('up', up)
    # print('n', n)
    return Yz

def init_x_P(init_angle = 0):

    P = np.eye(3, dtype=float)
    P[0, 0] = get_V()[0,0]
    P[1, 1] = get_V()[0,0]
    P[2, 2] = get_V()[1,1]

    x = np.zeros((3, 1))
    x[2,0] = init_angle
    return x, P

def get_V():
    sigma_x = 0.5/3
    sigma_psi = ((0/3)*np.pi/180)

    V = np.asarray([[sigma_x**2, 0],
                    [0, sigma_psi**2] ])

    return V

def apply_f(x, u):
    f11 = x[0,0] + np.cos(x[2,0])*u[0,0]
    f21 = x[1, 0] + np.sin(x[2,0])*u[0,0]
    f31 = get_sum_angles(x[2, 0], u[1,0])

    new_x = np.asarray([[f11],[f21],[f31]])
    return new_x

def get_Fx(x, u):
    Fx = np.asarray([[1, 0, -np.sin(x[2,0])*u[0,0]],
                     [0, 1, np.cos(x[2,0])*u[0,0]],
                     [0, 0, 1]])
    return Fx

def get_Fv(x, u):
    Fv = np.asarray([[np.cos(x[2,0]), 0],
                     [np.sin(x[2,0]), 0],
                     [0, 1]])
    return Fv


# def apply_h_plane(x, N):
#     d = np.linalg.norm(N)
#     a = N[0,0]/d
#     b = N[1,0]/d
#     c = N[2,0]/d

#     ux = a*(d + a*x[0,0] + b*x[1,0])
#     uy = b*(d + a*x[0,0] + b*x[1,0])
#     uz = c*(d + a*x[0,0] + b*x[1,0])

#     zp = np.asarray( [[np.cos(x[2,0])*ux + np.sin(x[2,0])*uy],
#                       [-np.sin(x[2,0])*ux + np.cos(x[2,0])*uy],
#                       [uz]])
#     return zp

# def apply_h_plane(x, N):
#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     d  = N[3,0]

#     dist_mov = x[0,0]**2+x[1,0]**2

#     zp = np.asarray( [[np.cos(x[2,0])*na + np.sin(x[2,0])*nb],
#                       [-np.sin(x[2,0])*na + np.cos(x[2,0])*nb],
#                       [nc],
#                       [(na*dist_mov+nb*dist_mov+d)]])
#     if zp[3,0] < 0:
#         print("Negativou")
#         zp[0,0] = -zp[0,0]
#         zp[1,0] = -zp[1,0]
#         zp[2,0] = -zp[2,0]
#         zp[3,0] = -zp[3,0]
#     return zp


# def apply_h_plane(x, N):
#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     d = N[3,0]

#     R = get_rotation_matrix_bti([0, 0, x[2,0]])
#     H = np.asarray([[R[0,0], R[0,1], R[0,2], x[0,0]],
#                     [R[1,0], R[1,1], R[1,2], x[1,0]],
#                     [R[2,0], R[2,1], R[2,2], 0],
#                     [0, 0, 0, 1],])

#     zp = np.dot(H.T,N)
#     return zp

# def apply_h_plane(x1, N1):
#     N = copy.deepcopy(N1)
#     x = copy.deepcopy(x1)
#     N = normalize_plane_feature(N)
#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     d  = N[3,0]


#     zp = np.asarray( [[np.cos(x[2,0])*na + np.sin(x[2,0])*nb],
#                       [-np.sin(x[2,0])*na + np.cos(x[2,0])*nb],
#                       [nc],
#                       [(na*x[0,0]+nb*x[1,0]+d)]])
#     zp = normalize_plane_feature(zp)
    # return zp


# def apply_h_plane(x, N):
#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     nd  = N[3,0]

#     n = np.dot(get_rotation_matrix_bti([0, 0, x[2,0]]).T, [na, nb, nc])

#     d = (-na*x[0,0]-nb*x[1,0]+nd)


#     zp = np.asarray([[n[0], n[1], n[2], d]]).T
#     return zp

# def apply_h_plane(x1, N1):
#     N = copy.deepcopy(N1)
#     x = copy.deepcopy(x1)
#     N = normalize_plane_feature(N)

#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     nd  = N[3,0]

#     n = np.dot(get_rotation_matrix_bti([0, 0, x[2,0]]).T, [na, nb, nc])

#     d = (-na*x[0,0]-nb*x[1,0]+nd)


#     zp = np.asarray([[n[0], n[1], n[2], d]]).T
#     zp = normalize_plane_feature(zp)
#     return zp


def apply_h_plane(x1, N1):
    N = copy.deepcopy(N1)
    x = copy.deepcopy(x1)
    N = normalize_plane_feature(N)
    na = N[0,0]
    nb = N[1,0]
    nc = N[2,0]
    nd  = N[3,0]

    n = np.asarray([[na*np.cos(x[2,0]) + nb*np.sin(x[2,0])],
                    [-na*np.sin(x[2,0]) + nb*np.cos(x[2,0])],
                    [nc],
                    [na*x[0,0] + nb*x[1,0] + nd]])

    zp = normalize_plane_feature(n)
    return zp




# def get_Hxv_plane(x, N):
#     d = np.linalg.norm(N)
#     a = N[0,0]/d
#     b = N[1,0]/d
#     c = N[2,0]/d

#     ux = a*(d+a*x[0,0] + b*x[1,0])
#     uy = b*(d+a*x[0,0] + b*x[1,0])
#     uz = c*(d+a*x[0,0] + b*x[1,0])

#     hx11 = a*a*np.cos(x[2,0]) + a*b*np.sin(x[2,0])
#     hx12 = a*b*np.cos((x[2,0])) + b*b*np.sin((x[2,0]))
#     hx13 = -np.sin(x[2,0])*ux + np.cos(x[2,0])*uy 

#     hx21 = -a*a*np.sin(x[2,0]) + a*b* np.cos(x[2,0])
#     hx22 = -a*b*np.sin(x[2,0]) + b*b* np.cos(x[2,0])
#     hx23 = -np.cos(x[2,0]) *ux - np.sin(x[2,0])*uy

#     hx31 = c*a
#     hx32 = b*c
#     hx33 = 0

#     Hx = np.asarray([[hx11, hx12, hx13],
#                      [hx21, hx22, hx23],
#                      [hx31, hx32, hx33]])
#     return Hx


# def get_Hxv_plane(x1, N1):
#     N = copy.deepcopy(N1)
#     x = copy.deepcopy(x1)
#     N = normalize_plane_feature(N)

#     na = N[0,0]#*np.sign(np.cos(x[2,0]))
#     nb = N[1,0]#*np.sign(np.sin(-x[2,0]))
#     nc = N[2,0]
#     d = N[3,0]

#     hx11 = 0
#     hx12 = 0
#     hx13 = -np.sin(x[2,0])*na + np.cos(x[2,0])*nb


#     hx21 = 0 
#     hx22 = 0
#     hx23 = -np.cos(x[2,0])*na - np.sin(x[2,0])*nb


#     hx31 = 0
#     hx32 = 0
#     hx33 = 0

#     hx41 = na*np.cos(x[2,0]) + nb*np.sin(x[2,0])
#     hx42 = na*np.sin(x[2,0]) + nb*np.cos(x[2,0]) 
#     hx43 = 0


#     Hx = np.asarray([[hx11, hx12, hx13],
#                      [hx21, hx22, hx23],
#                      [hx31, hx32, hx33],
#                      [hx41, hx42, hx43]])
#     return Hx


def get_Hxv_plane(x_state1, N1):
    N = copy.deepcopy(N1)
    x_state = copy.deepcopy(x_state1)

    na = N[0,0]
    nb = N[1,0]
    nc = N[2,0]
    nd = N[3,0]

    x = x_state[0,0]
    y = x_state[1,0]
    theta = x_state[2,0]

    Hx =    np.asarray([[  0,  0,   nb*np.cos(theta) - na*np.sin(theta)],
             [  0,  0, - na*np.cos(theta) - nb*np.sin(theta)],
             [  0,  0,                               0],
             [ na, nb,                               0]])

    return Hx




# def get_Hxv_plane(x, N):
#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     d = N[3,0]

#     hx11 = 0
#     hx12 = 0
#     hx13 = -nb


#     hx21 = 0
#     hx22 = 0
#     hx23 = na


#     hx31 = 0
#     hx32 = 0
#     hx33 = 0

#     hx41 = na
#     hx42 = nb
#     hx43 = 0


#     Hx = np.asarray([[hx11, hx12, hx13],
#                      [hx21, hx22, hx23],
#                      [hx31, hx32, hx33],
#                      [hx41, hx42, hx43]])
#     return Hx


# def get_Hxp_plane(x, N):
#     d = np.linalg.norm(N)
#     a = N[0,0]/d
#     b = N[1,0]/d
#     c = N[2,0]/d

#     dh1dnx =  (np.cos(x[2,0])*(- 2*x[0,0]*a**3 - 2*b*x[1,0]*a**2 + 2*x[0,0]*a + d + b*x[1,0]))/d - (b*np.sin(x[2,0])*(2*x[0,0]*a**2 + 2*b*x[1,0]*a - x[0,0]))/d
#     dh1dny =  (np.sin(x[2,0])*(- 2*x[1,0]*b**3 - 2*a*x[0,0]*b**2 + 2*x[1,0]*b + d + a*x[0,0]))/d - (a*np.cos(x[2,0])*(2*x[1,0]*b**2 + 2*a*x[0,0]*b - x[1,0]))/d
#     dh1dnz = -(2*c*(a*np.cos(x[2,0]) + b*np.sin(x[2,0]))*(a*x[0,0] + b*x[1,0]))/d

#     dh2dnx = -(np.sin(x[2,0])*(- 2*x[0,0]*a**3 - 2*b*x[1,0]*a**2 + 2*x[0,0]*a + d + b*x[1,0]))/d - (b*np.cos(x[2,0])*(2*x[0,0]*a**2 + 2*b*x[1,0]*a - x[0,0]))/d
#     dh2dny =  (np.cos(x[2,0])*(- 2*x[1,0]*b**3 - 2*a*x[0,0]*b**2 + 2*x[1,0]*b + d + a*x[0,0]))/d + (a*np.sin(x[2,0])*(2*x[1,0]*b**2 + 2*a*x[0,0]*b - x[1,0]))/d
#     dh2dnz = -(2*c*(b*np.cos(x[2,0]) - a*np.sin(x[2,0]))*(a*x[0,0] + b*x[1,0]))/d

#     dh3dnx = -(c*(2*x[0,0]*a**2 + 2*b*x[1,0]*a - x[0,0]))/d
#     dh3dny = -(c*(2*x[1,0]*b**2 + 2*a*x[0,0]*b - x[1,0]))/d
#     dh3dnz =  (d + a*x[0,0] + b*x[1,0] - 2*a*c**2*x[0,0] - 2*b*c**2*x[1,0])/d

#     Hxp = np.asarray([[dh1dnx, dh1dny, dh1dnz],
#                      [dh2dnx, dh2dny, dh2dnz],
#                      [dh3dnx, dh3dny, dh3dnz]])
#     return Hxp


# def get_Hxp_plane(x1, N1):
#     N = copy.deepcopy(N1)
#     x = copy.deepcopy(x1)
#     N = normalize_plane_feature(N)

#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     d = N[3,0]

#     dh1dna = np.cos(x[2,0])
#     dh1dnb = np.sin(x[2,0])
#     dh1dnc = 0
#     dh1dd  = 0

#     dh2dna = -np.sin(x[2,0])
#     dh2dnb = np.cos(x[2,0])
#     dh2dnc = 0
#     dh2dd  = 0

#     dh3dna = 0
#     dh3dnb = 0
#     dh3dnc = 1
#     dh3dd  = 0

#     dh4dna = -x[0,0]*np.sin(x[2,0]) + x[1,0]*np.cos(x[2,0])
#     dh4dnb = x[0,0]*np.cos(x[2,0]) - x[1,0]*np.sin(x[2,0])
#     dh4dnc = 0
#     dh4dd  = 1

#     Hxp = np.asarray([[dh1dna, dh1dnb, dh1dnc, dh1dd],
#                      [dh2dna, dh2dnb, dh2dnc, dh2dd],
#                      [dh3dna, dh3dnb, dh3dnc, dh3dd],
#                      [dh4dna, dh4dnb, dh4dnc, dh4dd]])
#     return Hxp


def get_Hxp_plane(x_state1, N1):
    N = copy.deepcopy(N1)
    x_state = copy.deepcopy(x_state1)

    na = N[0,0]
    nb = N[1,0]
    nc = N[2,0]
    nd = N[3,0]

    x = x_state[0,0]
    y = x_state[1,0]
    theta = x_state[2,0]

    Hz =    np.asarray([[  np.cos(theta), np.sin(theta), 0, 0],
             [ -np.sin(theta), np.cos(theta), 0, 0],
             [           0,          0, 1, 0],
             [           x,          y, 0, 1]])

    return Hz







# def apply_g_plane(x, Zp):


#     Zp = np.dot(get_rotation_matrix_bti([0, 0, x[2,0]]), Zp)

#     eta = np.asarray([[x[0,0]], [x[1,0]], [0]])
#     # print('eta: ', eta)
#     # eta = np.dot(get_rotation_matrix_bti([0, 0, x[2,0]]), eta)
#     # print('eta2: ', eta)

#     corre = (np.dot(eta.T, Zp)/(np.linalg.norm(Zp)**2))
#     u = Zp - corre*Zp

#     # N = np.asarray( [[np.cos(x[2,0])*u[0,0] - np.sin(x[2,0])*u[1,0]],
#     #                  [np.sin(x[2,0])*u[0,0] + np.cos(x[2,0])*u[1,0]],
#     #                  [u[2,0]]])
#     return u

##### CORRETO
# def apply_g_plane(x1, Zp1):
#     Zp= copy.deepcopy(Zp1)
#     x = copy.deepcopy(x1)
#     Zp = normalize_plane_feature(Zp)
#     nam = Zp[0,0]
#     nbm = Zp[1,0]
#     ncm = Zp[2,0]
#     dm  = Zp[3,0]

#     n = np.dot(get_rotation_matrix_bti([0, 0, x[2,0]]), [nam, nbm, ncm])

#     #d = -nam*x[0,0]-nbm*x[1,0]+dm
#     #x_inv = np.dot(get_rotation_matrix_bti([0, 0, x[2,0]]).T, [nam*x[0,0], nbm*x[1,0], 0])
#     d = -n[0]*x[0,0]-n[1]*x[1,0] + dm

#     z = np.asarray([[n[0], n[1], n[2], d]]).T
#     z = normalize_plane_feature(z)
#     return z


def apply_g_plane(x1, Zp1):
    Zp= copy.deepcopy(Zp1)
    x = copy.deepcopy(x1)
    Zp = normalize_plane_feature(Zp)
    na = Zp[0,0]
    nb = Zp[1,0]
    nc = Zp[2,0]
    nd  = Zp[3,0]

    n = np.asarray([[na*np.cos(x[2,0]) - nb*np.sin(x[2,0]) ],
                    [na*np.sin(x[2,0]) + nb*np.cos(x[2,0])],
                    [nc],
                    [-(na*np.cos(x[2,0]) - nb*np.sin(x[2,0]))*x[0,0]  -(na*np.sin(x[2,0]) + nb*np.cos(x[2,0]))*x[1,0] + nd]])

    zp = normalize_plane_feature(n)
    return zp


# def apply_g_plane(x1, N1):
#     N = copy.deepcopy(N1)
#     x = copy.deepcopy(x1)
#     N = normalize_plane_feature(N)
#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     d  = N[3,0]


#     zp = np.asarray( [[na*np.cos(x[2,0]) - nb*np.sin(x[2,0])],
#                       [na*np.sin(x[2,0]) + nb*np.cos(x[2,0])],
#                       [nc],
#                       [na*(-x[0,0]*np.cos(x[2,0])-x[1,0]*np.sin(x[2,0])) + nb*(x[0,0]*np.sin(x[2,0]) - x[1,0]*np.cos(x[2,0]) ) + d]])
#     zp = normalize_plane_feature(zp)
#     return zp

# def apply_g_plane(x, N):
#     N = copy.deepcopy(N)
#     x = copy.deepcopy(x)
#     N = normalize_plane_feature(N)
#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     d = N[3,0]



#     R = get_rotation_matrix_bti([0, 0, x[2,0]])
#     H = np.asarray([[R[0,0], R[0,1], R[0,2], x[0,0]],
#                     [R[1,0], R[1,1], R[1,2], x[1,0]],
#                     [R[2,0], R[2,1], R[2,2], 0],
#                     [0, 0, 0, 1],])

#     H_inv = np.linalg.inv(H)
#     zp = np.dot(H_inv.T,N)
#     zp = normalize_plane_feature(zp)
#     return zp

# def get_Gx_plane(x, Zp):
#     d = np.linalg.norm(Zp)
#     a = Zp[0,0]/d
#     b = Zp[1,0]/d
#     c = Zp[2,0]/d

#     dg1dx = (d**2*(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))**2)/(d**2*(a**2 + b**2 + c**2))**(1/2)
#     dg1dy = (d**2*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(a*np.cos(x[2,0]) - b*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2)
#     dg1dpsi = - a*d*np.sin(x[2,0]) - b*d*np.cos(x[2,0]) - (d**2*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) - (d**2*(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))*(b*x[0,0]*np.cos(x[2,0]) - a*x[1,0]*np.cos(x[2,0]) + a*x[0,0]*np.sin(x[2,0]) + b*x[1,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2)
 
#     dg2dx = (d**2*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(a*np.cos(x[2,0]) - b*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2)
#     dg2dy = (d**2*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))**2)/(d**2*(a**2 + b**2 + c**2))**(1/2)
#     dg2dpsi = a*d*np.cos(x[2,0]) - b*d*np.sin(x[2,0]) - (d**2*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(b*x[0,0]*np.cos(x[2,0]) - a*x[1,0]*np.cos(x[2,0]) + a*x[0,0]*np.sin(x[2,0]) + b*x[1,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) + (d**2*(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2)

#     dg3dx = (c*d**2*(a*np.cos(x[2,0]) - b*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2)
#     dg3dy = (c*d**2*(b*np.cos(x[2,0]) + a*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2)
#     dh3dpsi = -(c*d**2*(b*x[0,0]*np.cos(x[2,0]) - a*x[1,0]*np.cos(x[2,0]) + a*x[0,0]*np.sin(x[2,0]) + b*x[1,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2)

#     Gxp = np.asarray([[dg1dx, dg1dy, dg1dpsi],
#                      [dg2dx, dg2dy, dg2dpsi],
#                      [dg3dx, dg3dy, dh3dpsi]])
#     return Gxp

# def get_Gx_plane(x1, N1):
#     N = copy.deepcopy(N1)
#     x = copy.deepcopy(x1)
#     N = normalize_plane_feature(N)
#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     d = N[3,0]

#     hx11 = 0
#     hx12 = 0
#     hx13 = -np.sin(x[2,0])*na - np.cos(x[2,0])*nb


#     hx21 = 0
#     hx22 = 0
#     hx23 = np.cos(x[2,0])*na - np.sin(x[2,0])*nb


#     hx31 = 0
#     hx32 = 0
#     hx33 = 0

#     hx41 = -(na*np.cos(x[2,0]) - nb*np.sin(x[2,0]))
#     hx42 = (na*np.sin(x[2,0]) + nb*np.cos(x[2,0]))
#     hx43 = x[0,0]*(np.sin(x[2,0])*na +np.cos(x[2,0])*nb) + x[1,0]*(-na*np.cos(x[2,0])+nb*np.sin(x[2,0]))





#     Hx = np.asarray([[hx11, hx12, hx13],
#                      [hx21, hx22, hx23],
#                      [hx31, hx32, hx33],
#                      [hx41, hx42, hx43]])
#     return Hx



def get_Gx_plane(x_state1, N1):
    N = copy.deepcopy(N1)
    x_state = copy.deepcopy(x_state1)

    na = N[0,0]
    nb = N[1,0]
    nc = N[2,0]
    nd = N[3,0]

    x = x_state[0,0]
    y = x_state[1,0]
    theta = x_state[2,0]

    Gx =    np.asarray([[                             0,                               0,                                       - nb*np.cos(theta) - na*np.sin(theta)],
                         [                             0,                               0,                                         na*np.cos(theta) - nb*np.sin(theta)],
                         [                             0,                               0,                                                                     0],
                         [ nb*np.sin(theta) - na*np.cos(theta), - nb*np.cos(theta) - na*np.sin(theta), x*(nb*np.cos(theta) + na*np.sin(theta)) - y*(na*np.cos(theta) - nb*np.sin(theta))]])

    return Gx

# def get_Gz_plane(x, Zp):
#     d = np.linalg.norm(Zp)
#     a = Zp[0,0]/d
#     b = Zp[1,0]/d
#     c = Zp[2,0]/d

#     dg1dnx = np.cos(x[2,0]) + (d*(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))*(x[0,0]*np.cos(x[2,0]) + x[1,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) + (d*np.cos(x[2,0])*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) - (a*d**3*(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(3/2)
#     dg1dny = (d*(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))*(x[1,0]*np.cos(x[2,0]) - x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) - np.sin(x[2,0]) - (d*np.sin(x[2,0])*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) - (b*d**3*(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(3/2)
#     dg1dnz = -(c*d**3*(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(3/2)
    
#     dg2dnx = np.sin(x[2,0]) + (d*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(x[0,0]*np.cos(x[2,0]) + x[1,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) + (d*np.sin(x[2,0])*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) - (a*d**3*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(3/2)
#     dg2dny = np.cos(x[2,0]) + (d*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(x[1,0]*np.cos(x[2,0]) - x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) + (d*np.cos(x[2,0])*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) - (b*d**3*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(3/2)
#     dg2dnz = -(c*d**3*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(3/2)

#     dg3dnx = (c*d**3*(b**2*x[0,0]*np.cos(x[2,0]) + c**2*x[0,0]*np.cos(x[2,0]) + b**2*x[1,0]*np.sin(x[2,0]) + c**2*x[1,0]*np.sin(x[2,0]) - a*b*x[1,0]*np.cos(x[2,0]) + a*b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(3/2)
#     dg3dny = -(c*d**3*(a**2*x[0,0]*np.sin(x[2,0]) - c**2*x[1,0]*np.cos(x[2,0]) - a**2*x[1,0]*np.cos(x[2,0]) + c**2*x[0,0]*np.sin(x[2,0]) + a*b*x[0,0]*np.cos(x[2,0]) + a*b*x[1,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(3/2)
#     dg3dnz = (d*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(1/2) - (c**2*d**3*(a*x[0,0]*np.cos(x[2,0]) + b*x[1,0]*np.cos(x[2,0]) + a*x[1,0]*np.sin(x[2,0]) - b*x[0,0]*np.sin(x[2,0])))/(d**2*(a**2 + b**2 + c**2))**(3/2) + 1

#     Gz = np.asarray([[dg1dnx, dg1dny, dg1dnz],
#                      [dg2dnx, dg2dny, dg2dnz],
#                      [dg3dnx, dg3dny, dg3dnz]])
#     return Gz

# def get_Gz_plane(x1, N1):
#     N = copy.deepcopy(N1)
#     x = copy.deepcopy(x1)
#     N = normalize_plane_feature(N)
#     na = N[0,0]
#     nb = N[1,0]
#     nc = N[2,0]
#     d  = N[3,0]

#     dh1dna = np.cos(x[2,0])
#     dh1dnb = -np.sin(x[2,0])
#     dh1dnc = 0
#     dh1dd  = 0

#     dh2dna = np.sin(x[2,0])
#     dh2dnb = np.cos(x[2,0])
#     dh2dnc = 0
#     dh2dd  = 0

#     dh3dna = 0
#     dh3dnb = 0
#     dh3dnc = 1
#     dh3dd  = 0

#     dh4dna = -x[0,0]*np.cos(x[2,0]) - x[1,0]*np.sin(x[2,0])
#     dh4dnb = x[0,0]*np.sin(x[2,0]) - x[1,0]*np.cos(x[2,0])
#     dh4dnc = 0
#     dh4dd  = 1

#     Hxp = np.asarray([[dh1dna, dh1dnb, dh1dnc, dh1dd],
#                      [dh2dna, dh2dnb, dh2dnc, dh2dd],
#                      [dh3dna, dh3dnb, dh3dnc, dh3dd],
#                      [dh4dna, dh4dnb, dh4dnc, dh4dd]])
#     return Hxp



def get_Gz_plane(x_state1, N1):
    N = copy.deepcopy(N1)
    x_state = copy.deepcopy(x_state1)

    na = N[0,0]
    nb = N[1,0]
    nc = N[2,0]
    nd = N[3,0]

    x = x_state[0,0]
    y = x_state[1,0]
    theta = x_state[2,0]

    Gx =    np.asarray([[                    np.cos(theta),                 -np.sin(theta), 0, 0],
                         [                    np.sin(theta),                  np.cos(theta), 0, 0],
                         [                             0,                           0, 1, 0],
                         [ - x*np.cos(theta) - y*np.sin(theta), x*np.sin(theta) - y*np.cos(theta), 0, 1]])


    return Gx





def get_W_plane():
    sigma_x = 0.1/3
    sigma_y = 0.1/3
    sigma_z = 0.1/3
    sigma_d = 0.1/3

    W = np.asarray([[sigma_x**2, 0, 0, 0],
                    [0, sigma_y**2, 0, 0],
                    [0, 0, sigma_z**2, 0],
                    [0, 0, 0, sigma_d**2] ])
    return W

def get_Hw_plane():
    Hw = np.asarray([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    return Hw



#############################################################################################
# POINT FEATURE - CYLINDER AND CUBOID
#############################################################################################



def apply_h_point(x, Z):

    zpx = np.cos(x[2,0])*(Z[0,0] - x[0,0]) + np.sin(x[2,0])*(Z[1,0] - x[1,0])
    zpy = -np.sin(x[2,0])*(Z[0,0] - x[0,0]) + np.cos(x[2,0])*(Z[1,0] - x[1,0])
    zpz = Z[2,0]

    zp = np.asarray( [[zpx],
                      [zpy],
                      [zpz]])
    return zp

def apply_g_point(x, C):
    zx = x[0,0] + C[0,0]*np.cos(x[2,0]) - C[1,0]*np.sin(x[2,0])
    zy = x[1,0] + C[1,0]*np.cos(x[2,0]) + C[0,0]*np.sin(x[2,0])
    zz = C[2,0]

    Z = np.asarray( [[zx],
                     [zy],
                     [zz]])
    return Z


def get_Hxv_point(x, Z):

    hx11 = -np.cos(x[2,0])
    hx12 = -np.sin(x[2,0])
    hx13 = np.sin(x[2,0])*(x[0,0] - Z[0,0]) - np.cos(x[2,0])*(x[1,0] - Z[1,0])

    hx21 = np.sin(x[2,0])
    hx22 = -np.cos(x[2,0])
    hx23 = np.cos(x[2,0])*(x[0,0] - Z[0,0]) + np.sin(x[2,0])*(x[1,0] - Z[1,0])

    hx31 = 0
    hx32 = 0
    hx33 = 0

    Hx = np.asarray([[hx11, hx12, hx13],
                     [hx21, hx22, hx23],
                     [hx31, hx32, hx33]])
    return Hx


def get_Hxp_point(x, N):

    dh1dnx =  np.cos(x[2,0])
    dh1dny =  np.sin(x[2,0])
    dh1dnz = 0

    dh2dnx = -np.sin(x[2,0])
    dh2dny = np.cos(x[2,0])
    dh2dnz = 0

    dh3dnx = 0
    dh3dny = 0
    dh3dnz = 1

    Hxp = np.asarray([[dh1dnx, dh1dny, dh1dnz],
                     [dh2dnx, dh2dny, dh2dnz],
                     [dh3dnx, dh3dny, dh3dnz]])
    return Hxp


def get_Gx_point(x, C):

    dg1dx = 1
    dg1dy = 0
    dg1dpsi = - C[1,0]*np.cos(x[2,0]) - C[0,0]*np.sin(x[2,0])

    dg2dx = 0
    dg2dy = 1
    dg2dpsi = C[0,0]*np.cos(x[2,0]) - C[1,0]*np.sin(x[2,0])

    dg3dx = 0
    dg3dy = 0
    dh3dpsi = 0

    Gxp = np.asarray([[dg1dx, dg1dy, dg1dpsi],
                     [dg2dx, dg2dy, dg2dpsi],
                     [dg3dx, dg3dy, dh3dpsi]])
    return Gxp

def get_Gz_point(x, C):


    dg1dnx = np.cos(x[2,0])
    dg1dny = -np.sin(x[2,0])
    dg1dnz = 0
    
    dg2dnx = np.sin(x[2,0])
    dg2dny = np.cos(x[2,0])
    dg2dnz = 0

    dg3dnx = 0
    dg3dny = 0
    dg3dnz = 1

    Gz = np.asarray([[dg1dnx, dg1dny, dg1dnz],
                     [dg2dnx, dg2dny, dg2dnz],
                     [dg3dnx, dg3dny, dg3dnz]])
    return Gz

def get_W_point():
    sigma_x = 0.1/3
    sigma_y = 0.1/3
    sigma_z = 0.1/3

    W = np.asarray([[sigma_x**2, 0, 0],
                    [0, sigma_y**2, 0],
                    [0, 0, sigma_z**2] ])
    return W

def get_Hw_point():
    Hw = np.asarray([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    return Hw