import numpy as np

def init_x_P():
    # 7 paredes = 21 parãmetros
    numParedes = 0
    dim = 3 + numParedes*3

    P = np.eye(dim, dtype=int)*99
    P[0, 0] = 0
    P[1, 1] = 0
    P[2, 2] = 0

    x = np.zeros((dim, 1))
    return x, P

def get_V():
    sigma_x = 0.1/3
    sigma_psi = (15*np.pi/180)/3

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
    a = N[0]/d
    b = N[1]/d
    c = N[2]/d

    ux = a*(d + a*x[0] + b*x[1])
    uy = b*(d + a*x[0] + b*x[1])
    uz = c*(d + a*x[0] + b*x[1])

    zp = np.asarray( [np.cos(x[2])*ux + np.sin(x[2])*uy,
                      -np.sin(x[2])*ux + np.cos(x[2])*uy,
                      uz])
    return zp


def apply_g(x, Zp):
    d = np.linalg.norm(Zp)
    a = Zp[0]/d
    b = Zp[1]/d
    c = Zp[2]/d

    ux = a*(d - a*x[0] - b*x[1])
    uy = b*(d - a*x[0] - b*x[1])
    uz = c*(d - a*x[0] - b*x[1])

    N = np.asarray( [np.cos(x[2])*ux - np.sin(x[2])*uy,
                      +np.sin(x[2])*ux + np.cos(x[2])*uy,
                      uz])
    return N




def get_Hx_plane(x, N):
    d = np.linalg.norm(N)
    a = N[0]/d
    b = N[1]/d
    c = N[2]/d

    ux = a*(d+a*x[0] + b*x[1])
    uy = b*(d+a*x[0] + b*x[1])
    uz = c*(d+a*x[0] + b*x[1])

    hx11 = a*a*np.cos(x[2]) + a*b*np.sin(x[2])
    hx12 = a*b*np.cos((x[2])) + b*b*np.sin((x[2]))
    hx13 = -np.sin(x[2])*ux + np.cos(x[2])*uy 

    hx21 = -a*a*np.sin(x[2]) + a*b* np.cos(x[2])
    hx22 = -a*b*np.sin(x[2]) + b*b* np.cos(x[2])
    hx23 = -np.cos(x[2]) *ux - np.sin(x[2])*uy

    hx31 = c*a
    hx32 = b*c
    hx33 = 0

    Hx = np.asarray([[hx11, hx12, hx13],
                     [hx21, hx22, hx23],
                     [hx31, hx32, hx33]])
    return Hx

def get_Hw_plane():
    Hw = np.asarray([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    return Hw