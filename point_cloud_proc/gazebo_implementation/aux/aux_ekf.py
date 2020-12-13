import numpy as np
import pickle

class ekf:

    def __init__(self):
        self.x_m, self.P_m  = init_x_P()
        self.x_p, self.P_p  = init_x_P()

        self.x_p_list = []
        self.P_p_list = []
        self.num_total_features = {'plane':0}


    def propagate(self, u):
        x_m_last = self.x_m[:3,:]
        P_m_last = self.P_m[:3,:3]
        self.x_p = self.x_m
        self.P_p = self.P_m
        print('u ',u )
        #print("self.P_m: ", self.P_m)
        #print("x_m_last: \n",x_m_last)
        #print("P_m_last: \n",P_m_last)
        #print("u: \n",u)

        Fx = get_Fx(x_m_last, u)
        Fv = get_Fv(x_m_last, u)
        V  = get_V()

        print("antes: \n", x_m_last)
        new_x_p = apply_f(x_m_last, u) # Propagation
        print("depois: \n", new_x_p)
        #print("x_p: \n",x_p)
        new_P_p = Fx @ P_m_last @ Fx.T +  Fv @ V @ Fv.T
        #print("new_P_p: ", new_P_p)



        self.x_p[:3,:] = new_x_p
        print("depois2: \n", self.x_p[:3,:])
        self.P_p[:3,:3] = new_P_p
        #print("self.P_p: ", self.P_p)

        self.P_m = self.P_p
        self.x_m = self.x_p

        self.x_p_list.append(self.x_p.copy())
        print("gravandoisso: \n", self.x_p_list)
        self.P_p_list.append(self.P_p.copy())

        # print("f: ",self.x_p)
        # print("P: ",self.P_p)
        nsalva = {}
        nsalva['x_p_list'] =  self.x_p_list
        nsalva['P_p_list'] =  self.P_p_list

        f = open('ekf.pckl', 'wb')
        pickle.dump(nsalva, f)
        f.close()

    def add_plane(self, Z):
        i_plano = self.num_total_features['plane'] # pega id do próximo plano
        self.num_total_features['plane'] = self.num_total_features['plane']+1 # soma contador de planos

        

        Gx = get_Gx_plane(self.x_m, Z)
        Gz = get_Gz_plane(self.x_m, Z)
        Yz = get_Yz(Gx, Gz, self.P_m)
        # print('Yz ', Yz)

        N = apply_g(self.x_m, Z)
        self.x_m = np.vstack((self.x_m, N))
        # print("New Feature: ", N)
        # print("New x: ", self.x_m)
        W = get_W()
        meio_bloco = np.block([[self.P_m,                                   np.zeros((self.P_m.shape[0], W.shape[1]))],
                               [np.zeros((W.shape[0], self.P_m.shape[1])),  W]])
        # print('meio_bloco: \n', meio_bloco)

        self.P_m = Yz @ meio_bloco @ Yz.T
        return i_plano


    def upload_plane(self, Z, id):
        Hxv = get_Hxv_plane(self.x_m, Z)
        Hxp = get_Hxp_plane(self.x_m, Z)
        Hx = get_Hx(Hxv, Hxp, id, self.P_m)
        Hw = get_Hw_plane()
        W = get_W()

        S = Hx @ self.P_m @ Hx.T + Hw @ W @ Hw.T
        K = self.P_m @ Hx.T @ np.linalg.inv(S)

        v = Z - apply_h(self.x_m, self.x_m[(3+id*3):(3+(id+1)*3)])

        self.x_m = self.x_m + K @ v
        self.P_m = self.P_m - K @ Hx @ self.P_m

        return self.x_m[(3+id*3):(3+(id+1)*3)]


def get_Hx(Hxv, Hxp, id, P_m):
    print('Hxv:\n',Hxv)
    print('Hxp:\n',Hxp)
    antes_p = np.zeros((Hxv.shape[0],id*3))
    depois_p = np.zeros((Hxv.shape[0],(P_m.shape[1]-(Hxv.shape[1]+id*Hxp.shape[1]+Hxp.shape[1]))))
    print('antes_p:\n',antes_p)
    print('depois_p:\n',depois_p)
    Hx = np.hstack((Hxv,antes_p,Hxp,depois_p))
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

def init_x_P():
    # 7 paredes = 21 parãmetros
    dim = 3

    P = np.eye(dim, dtype=int)
    P[0, 0] = 0
    P[1, 1] = 0
    P[2, 2] = 0

    x = np.zeros((dim, 1))
    return x, P

def get_V():
    sigma_x = 0.01/3
    sigma_psi = (1.5*np.pi/180)/3

    V = np.asarray([[sigma_x**2, 0],
                    [0, sigma_psi**2] ])
    return V

def apply_f(x, u):
    f11 = x[0,0] + np.cos(x[2,0])*u[0,0]
    f21 = x[1, 0] + np.sin(x[2,0])*u[0,0]
    f31 = x[2, 0] + u[1,0]

    new_x = np.asarray([[f11],[f21],[f31]])
    return new_x

def get_Fx(x, u):
    Fx = np.asarray([[1, 0, -np.sin(x[2,0])*u[0,0]],
                     [0, 1, np.cos(x[2,0])*u[0,0]],
                     [0, 0, 1]])
    return Fx

def get_Fv(x, u):
    Fv = np.asarray([[np.cos(x[2,0]), 0],
                     [np.cos(x[2,0]), 0],
                     [0, 1]])
    return Fv


def apply_h(x, N):
    d = np.linalg.norm(N)
    a = N[0,0]/d
    b = N[1,0]/d
    c = N[2,0]/d

    ux = a*(d + a*x[0,0] + b*x[1,0])
    uy = b*(d + a*x[0,0] + b*x[1,0])
    uz = c*(d + a*x[0,0] + b*x[1,0])

    zp = np.asarray( [[np.cos(x[2,0])*ux + np.sin(x[2,0])*uy],
                      [-np.sin(x[2,0])*ux + np.cos(x[2,0])*uy],
                      [uz]])
    return zp


def get_Hxv_plane(x, N):
    d = np.linalg.norm(N)
    a = N[0,0]/d
    b = N[1,0]/d
    c = N[2,0]/d

    ux = a*(d+a*x[0,0] + b*x[1,0])
    uy = b*(d+a*x[0,0] + b*x[1,0])
    uz = c*(d+a*x[0,0] + b*x[1,0])

    hx11 = a*a*np.cos(x[2,0]) + a*b*np.sin(x[2,0])
    hx12 = a*b*np.cos((x[2,0])) + b*b*np.sin((x[2,0]))
    hx13 = -np.sin(x[2,0])*ux + np.cos(x[2,0])*uy 

    hx21 = -a*a*np.sin(x[2,0]) + a*b* np.cos(x[2,0])
    hx22 = -a*b*np.sin(x[2,0]) + b*b* np.cos(x[2,0])
    hx23 = -np.cos(x[2,0]) *ux - np.sin(x[2,0])*uy

    hx31 = c*a
    hx32 = b*c
    hx33 = 0

    Hx = np.asarray([[hx11, hx12, hx13],
                     [hx21, hx22, hx23],
                     [hx31, hx32, hx33]])
    return Hx


def get_Hxp_plane(x, N):
    d = np.linalg.norm(N)
    a = N[0,0]/d
    b = N[1,0]/d
    c = N[2,0]/d

    dh1dnx =  (np.cos(x[2,0])*(- 2*x[0,0]*a**3 - 2*b*x[1,0]*a**2 + 2*x[0,0]*a + d + b*x[1,0]))/d - (b*np.sin(x[2,0])*(2*x[0,0]*a**2 + 2*b*x[1,0]*a - x[0,0]))/d
    dh1dny =  (np.sin(x[2,0])*(- 2*x[1,0]*b**3 - 2*a*x[0,0]*b**2 + 2*x[1,0]*b + d + a*x[0,0]))/d - (a*np.cos(x[2,0])*(2*x[1,0]*b**2 + 2*a*x[0,0]*b - x[1,0]))/d
    dh1dnz = -(2*c*(a*np.cos(x[2,0]) + b*np.sin(x[2,0]))*(a*x[0,0] + b*x[1,0]))/d

    dh2dnx = -(np.sin(x[2,0])*(- 2*x[0,0]*a**3 - 2*b*x[1,0]*a**2 + 2*x[0,0]*a + d + b*x[1,0]))/d - (b*np.cos(x[2,0])*(2*x[0,0]*a**2 + 2*b*x[1,0]*a - x[0,0]))/d
    dh2dny =  (np.cos(x[2,0])*(- 2*x[1,0]*b**3 - 2*a*x[0,0]*b**2 + 2*x[1,0]*b + d + a*x[0,0]))/d + (a*np.sin(x[2,0])*(2*x[1,0]*b**2 + 2*a*x[0,0]*b - x[1,0]))/d
    dh2dnz = -(2*c*(b*np.cos(x[2,0]) - a*np.sin(x[2,0]))*(a*x[0,0] + b*x[1,0]))/d

    dh3dnx = -(c*(2*x[0,0]*a**2 + 2*b*x[1,0]*a - x[0,0]))/d
    dh3dny = -(c*(2*x[1,0]*b**2 + 2*a*x[0,0]*b - x[1,0]))/d
    dh3dnz =  (d + a*x[0,0] + b*x[1,0] - 2*a*c**2*x[0,0] - 2*b*c**2*x[1,0])/d

    Hxp = np.asarray([[dh1dnx, dh1dny, dh1dnz],
                     [dh2dnx, dh2dny, dh2dnz],
                     [dh3dnx, dh3dny, dh3dnz]])
    return Hxp




def apply_g(x, Zp):
    d = np.linalg.norm(Zp)
    a = Zp[0,0]/d
    b = Zp[1,0]/d
    c = Zp[2,0]/d

    ux = a*(d - a*x[0,0] - b*x[1,0])
    uy = b*(d - a*x[0,0] - b*x[1,0])
    uz = c*(d - a*x[0,0] - b*x[1,0])

    N = np.asarray( [[np.cos(x[2,0])*ux - np.sin(x[2,0])*uy],
                     [np.sin(x[2,0])*ux + np.cos(x[2,0])*uy],
                     [uz]])
    return N

def get_Gx_plane(x, Zp):
    d = np.linalg.norm(Zp)
    a = Zp[0,0]/d
    b = Zp[1,0]/d
    c = Zp[2,0]/d

    dg1dx = - np.cos(x[2,0])*a**2 + b*np.sin(x[2,0])*a
    dg1dy = np.sin(x[2,0])*b**2 - a*np.cos(x[2,0])*b
    dg1dpsi = (b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(a*x[0,0] - d + b*x[1,0])

    dg2dx = - np.sin(x[2,0])*a**2 - b*np.cos(x[2,0])*a
    dg2dy = - np.cos(x[2,0])*b**2 - a*np.sin(x[2,0])*b
    dg2dpsi = -(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))*(a*x[0,0] - d + b*x[1,0])

    dg3dx = -a*c
    dg3dy = -b*c
    dh3dpsi = 0

    Gxp = np.asarray([[dg1dx, dg1dy, dg1dpsi],
                     [dg2dx, dg2dy, dg2dpsi],
                     [dg3dx, dg3dy, dh3dpsi]])
    return Gxp

def get_Gz_plane(x, Zp):
    d = np.linalg.norm(Zp)
    a = Zp[0,0]/d
    b = Zp[1,0]/d
    c = Zp[2,0]/d

    dg1dnx = (np.cos(x[2,0])*(2*x[0,0]*a**3 + 2*b*x[1,0]*a**2 - 2*x[0,0]*a + d - b*x[1,0]))/d - (b*np.sin(x[2,0])*(2*x[0,0]*a**2 + 2*b*x[1,0]*a - x[0,0]))/d
    dg1dny = (a*np.cos(x[2,0])*(2*x[1,0]*b**2 + 2*a*x[0,0]*b - x[1,0]))/d - (np.sin(x[2,0])*(2*x[1,0]*b**3 + 2*a*x[0,0]*b**2 - 2*x[1,0]*b + d - a*x[0,0]))/d
    dg1dnz = (2*c*(a*np.cos(x[2,0]) - b*np.sin(x[2,0]))*(a*x[0,0] + b*x[1,0]))/d
    
    dg2dnx = (np.sin(x[2,0])*(2*x[0,0]*a**3 + 2*b*x[1,0]*a**2 - 2*x[0,0]*a + d - b*x[1,0]))/d + (b*np.cos(x[2,0])*(2*x[0,0]*a**2 + 2*b*x[1,0]*a - x[0,0]))/d
    dg2dny = (np.cos(x[2,0])*(2*x[1,0]*b**3 + 2*a*x[0,0]*b**2 - 2*x[1,0]*b + d - a*x[0,0]))/d + (a*np.sin(x[2,0])*(2*x[1,0]*b**2 + 2*a*x[0,0]*b - x[1,0]))/d
    dg2dnz = (2*c*(b*np.cos(x[2,0]) + a*np.sin(x[2,0]))*(a*x[0,0] + b*x[1,0]))/d

    dg3dnx = (c*(2*x[0,0]*a**2 + 2*b*x[1,0]*a - x[0,0]))/d
    dg3dny = (c*(2*x[1,0]*b**2 + 2*a*x[0,0]*b - x[1,0]))/d
    dg3dnz = (d - a*x[0,0] - b*x[1,0] + 2*a*c**2*x[0,0] + 2*b*c**2*x[1,0])/d

    Gz = np.asarray([[dg1dnx, dg1dny, dg1dnz],
                     [dg2dnx, dg2dny, dg2dnz],
                     [dg3dnx, dg3dny, dg3dnz]])
    return Gz

def get_W():
    sigma_x = 0*0.01/3
    sigma_y = 0*0.01/3
    sigma_z = 0*0.01/3

    W = np.asarray([[sigma_x**2, 0, 0],
                    [0, sigma_y**2, 0],
                    [0, 0, sigma_z**2] ])
    return W

def get_Hw_plane():
    Hw = np.asarray([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    return Hw