import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from aux.aux_ekf import *
from aux.aux import *

def get_plane_surface(a, b, c, d):
    if a == 0:
        xs = np.linspace(-10, 10, 100)
        zs = np.linspace(-10, 10, 100)

        X, Z = np.meshgrid(xs, zs)
        Y = -(d + a*X)/b
        return X, Y, Z
    elif b == 0:
        ys = np.linspace(-10, 10, 100)
        zs = np.linspace(-10, 10, 100)

        Y, Z = np.meshgrid(ys, zs)
        X = -(d + b*Y)/a
        return X, Y, Z
    else:
        ys = np.linspace(-10, 10, 100)
        zs = np.linspace(-10, 10, 100)

        Y, Z = np.meshgrid(ys, zs)
        X = -(d + b*Y)/a
        return X, Y, Z




def get_plane_surface2(a, b, c, d):
    eq = [a, b, c, d]
    a,b,c,d = 0, 0, 1, 0

    x = np.linspace(-10/2,10/2,10)
    y = np.linspace(-10/2,10/2,10)

    X,Y = np.meshgrid(x,y)
    Z = -(d + a*X + b*Y) / c

    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    #print('positions: ', positions)
    ROT1 = get_rotation_matrix_bti([0, 0, 0])
    pos_rot = np.dot(ROT1,positions)
    center_point = np.asarray([0, 0, 0])
    center_point = np.stack([center_point]*pos_rot[0, :].shape[0],0)
    # print("Antes:", pos_rot[:, :3])
    pos_rot = pos_rot + center_point.T
    # print("Depois:", pos_rot[:, :3])

    deslocd = np.asarray([0, 0, -eq[3]])
    deslocd = np.stack([deslocd]*pos_rot[0, :].shape[0],0)
    pos_rot = pos_rot + deslocd.T
    # print("Depois deslocadod :", pos_rot[:, :3])
    ROT2 = get_rotationMatrix_from_vectors([0, 0, 1], [eq[0], eq[1], eq[2]])
    pos_rot = np.dot(ROT2,pos_rot)

    Xrot = pos_rot[0,:].reshape(X.shape)
    Yrot = pos_rot[1,:].reshape(Y.shape)
    # Zrot = np.stack([0]*Z.shape[0]*Z.shape[1],0).reshape(Z.shape)
    Zrot = pos_rot[2,:].reshape(Z.shape)


    return Xrot, Yrot, Zrot

    # ax4.plot(xx_p, xy_p , label='Posição')

p1 = np.asarray([[0, 1, 0, -2]])
#p1 = np.asarray([[-7.2900e-01, -6.8450e-01, -1.4000e-03,  6.3021e+00]])

pt1 = np.asarray([[1], [1], [0]])
pt2 = np.asarray([[-1], [0], [0]])
pt3 = np.asarray([[1], [-1], [0]])

print('pt1: ',pt1)


x = np.asarray([[-5],[-10],[np.pi]])

X, Y, Z = get_plane_surface(p1[0,0], p1[0,1], p1[0,2], p1[0,3])
fig = plt.figure()

X2, Y2, Z2 = get_plane_surface2(p1[0,0], p1[0,1], p1[0,2], p1[0,3])

gs = fig.add_gridspec(1,2)
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax1.plot_surface(X, Y, Z)
ax1.plot_surface(X2, Y2, Z2)
ax1.scatter(0, 0, marker='o', label='Local do robô')

ax1.scatter(pt1[0,0], pt1[1,0], marker='o', label='pt1')
ax1.scatter(pt2[0,0], pt2[1,0], marker='o', label='pt2')
ax1.scatter(pt3[0,0], pt3[1,0], marker='o', label='pt3')


N_robo = np.asarray([p1[0]]).T
print('N_robo: ', N_robo)

N_mundo = apply_g_plane(x , N_robo)
print('N_mundo: ', N_mundo)

pt1_mundo  = apply_g_point(x , pt1)
pt2_mundo  = apply_g_point(x , pt2)
pt3_mundo  = apply_g_point(x , pt3)

X, Y, Z = get_plane_surface(N_mundo[0,0], N_mundo[1,0], N_mundo[2,0], N_mundo[3,0])
X2, Y2, Z2 = get_plane_surface2(N_mundo[0,0], N_mundo[1,0], N_mundo[2,0], N_mundo[3,0])
ax2.plot_surface(X, Y, Z)
ax2.plot_surface(X2, Y2, Z2)

# X, Y, Z = get_plane_surface(N_mundo[0,0], N_mundo[1,0], N_mundo[2,0], N_mundo[3,0])
# ax2.plot_surface(X, Y, Z)

ax2.scatter(x[0,0], x[1,0], marker='o', label='Robo')
ax2.scatter(0, 0, marker='o', label='Robo')
ax2.scatter(pt1_mundo[0,0], pt1_mundo[1,0], marker='o', label='pt1')
ax2.scatter(pt2_mundo[0,0], pt2_mundo[1,0], marker='o', label='pt2')
ax2.scatter(pt3_mundo[0,0], pt3_mundo[1,0], marker='o', label='pt3')


# MÉTODO ""CORRETO""":
atual_loc = [x[0,0], x[1,0], 0]
atual_angulo = [0, 0, x[2,0]]
rotMatrix = get_rotation_matrix_bti(atual_angulo)
tranlation = atual_loc


pts = np.vstack((pt1.T,pt2.T,pt3.T))
print('pts: ', pts)

inlin = np.dot(pts, rotMatrix.T) + tranlation
print('era para estar certo: ', inlin)
pts_certo = np.vstack((pt1_mundo.T,pt2_mundo.T,pt3_mundo.T))
print('certo: ', pts_certo)

pt1_certotb = np.dot(rotMatrix, pt1)
pt2_certotb = np.dot(rotMatrix, pt2)
pt3_certotb = np.dot(rotMatrix, pt3)

pts_certo_tb = np.vstack((pt1_certotb.T,pt2_certotb.T,pt3_certotb.T))

print('certo tb: ', pts_certo)

N_robo2 = apply_h_plane(x , N_mundo)
print("N_robo2: ", N_robo2)
X, Y, Z = get_plane_surface(N_robo2[0,0], N_robo2[1,0], N_robo2[2,0], N_robo2[3,0])
ax1.plot_surface(X, Y, Z)

plt.show()