import pickle
from tkinter import *
from tkinter import ttk
import time
from threading import Thread
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from aux.aux import *

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def threaded_function(ekflist, prop):
    x_p_list = np.asarray(ekflist['x_p_list'])
    P_p_list = np.asarray(ekflist['P_p_list'])

    xx_p = x_p_list[:, 0, 0]
    xy_p = x_p_list[:, 1, 0]
    xtheta_p = x_p_list[:, 2, 0]

    ppx = P_p_list[:, 0, 0]
    ppy = P_p_list[:, 1, 1]
    pptheta = P_p_list[:, 2, 2]

    print('xx_p', xx_p)
    print('xy_p', xy_p)
    print('xtheta_p', xtheta_p)
    print('px', ppx)
    print('py', ppy)
    print('py', pptheta)

    fig = plt.figure()


    gs = fig.add_gridspec(3,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[:, 1], projection='3d')

    fig.suptitle('Estados')

    ax1.plot(xx_p, label='x_m')
    ax1.plot(xx_p+3*np.sqrt(ppx), label='+3*sqrt(p_p_x)')
    ax1.plot(xx_p-3*np.sqrt(ppx), label='-3*sqrt(p_p_x)')
    ax1.legend(loc="upper right")
    ax1.grid()

    ax2.plot(xy_p, label='y_m')
    ax2.plot(xy_p + 3*np.sqrt(ppy), label='+3*sqrt(p_p_y)')
    ax2.plot(xy_p - 3*np.sqrt(ppy), label='-3*sqrt(p_p_y)')
    ax2.legend(loc="upper right")
    ax2.grid()

    ax3.plot(xtheta_p, label='theta_m')
    ax3.plot(xtheta_p+3*np.sqrt(pptheta), label='+3*sqrt(p_p_theta)')
    ax3.plot(xtheta_p-3*np.sqrt(pptheta), label='-3*sqrt(p_p_theta)')
    ax3.legend(loc="upper right")
    ax3.grid()




    

    for key in prop: # key of proprety
        if(isinstance(prop[key], list)): # if is a list
            if((key == "planes")):
                for o in range(len(prop[key])):
                    print("EQ. DO PLANO:",prop[key][o]["equation"])
                    eq = prop[key][o]["equation"]
                    a,b,c,d = 0, 0, 1, 0

                    x = np.linspace(-prop[key][o]["width"],prop[key][o]["width"]/2,10)
                    y = np.linspace(-prop[key][o]["height"]/2,prop[key][o]["height"]/2,10)

                    X,Y = np.meshgrid(x,y)
                    Z = (d - a*X - b*Y) / c

                    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
                    #print('positions: ', positions)
                    ROT1 = get_rotation_matrix_bti([0, 0, prop[key][o]["rot_angle"]])
                    pos_rot = np.dot(ROT1,positions)
                    center_point = np.asarray([prop[key][o]["center2d"][0], prop[key][o]["center2d"][1], 0])
                    center_point = np.stack([center_point]*pos_rot[0, :].shape[0],0)
                    print("Antes:", pos_rot[:, :3])
                    pos_rot = pos_rot + center_point.T
                    print("Depois:", pos_rot[:, :3])

                    deslocd = np.asarray([0, 0, -eq[3]])
                    deslocd = np.stack([deslocd]*pos_rot[0, :].shape[0],0)
                    pos_rot = pos_rot + deslocd.T
                    print("Depois deslocadod :", pos_rot[:, :3])
                    ROT2 = get_rotationMatrix_from_vectors([0, 0, 1], [eq[0], eq[1], eq[2]])
                    pos_rot = np.dot(ROT2,pos_rot)

                    Xrot = pos_rot[0,:].reshape(X.shape)
                    Yrot = pos_rot[1,:].reshape(Y.shape)
                    Zrot = pos_rot[2,:].reshape(Z.shape)

                    # print('X', X[:].shape)
                    # print('Y', Y[:].shape)
                    # print('Z', Z[:].shape)
                    if(o == 0):
                        alp = 0.3
                    else:
                        alp = 1
                    surf = ax4.plot_surface(Xrot, Yrot, Zrot, alpha=alp)

                    # ax4.plot(xx_p, xy_p , label='Posição')

                    
                break
    ax4.legend(loc="upper right")
    ax4.grid()
    axisEqual3D(ax4)
    plt.show()
    print("xp: ", x_p_list)
    print("pp: ", P_p_list)


while True:

    f = open('ekf.pckl', 'rb')
    ekflist = pickle.load(f)
    f.close()

    f = open('feat.pckl', 'rb')
    obj = pickle.load(f)
    f.close()

    threaded_function(ekflist, obj)

    time.sleep(0.1)
    


