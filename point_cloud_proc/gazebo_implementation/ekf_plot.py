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
    x_p_list_total = ekflist['x_p_list']
    print(x_p_list_total)
    P_p_list_total = ekflist['P_p_list']
    x_p_list = []
    P_p_list = []
    for x_p in x_p_list_total:
        x_p_list.append(x_p[:3, :])
    x_p_list = np.asarray(x_p_list)


    for p_p in P_p_list_total:
        P_p_list.append(p_p[:3, :3])
    P_p_list = np.asarray(P_p_list)

    print('x_p_list: \n',x_p_list)
    #print('P_p_list: \n',x_p_list)

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

    fig = plt.figure(figsize=(14,7))


    gs = fig.add_gridspec(4,3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[:2, 2], projection='3d')
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[2, 2])
    ax7 = fig.add_subplot(gs[:2, 1])
    ax8 = fig.add_subplot(gs[3, :])

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
                    # print("EQ. DO PLANO:",prop[key][o]["equation"])
                    eq = prop[key][o]["equation"]
                    a,b,c,d = 0, 0, 1, 0

                    x = np.linspace(-prop[key][o]["width"]/2,prop[key][o]["width"]/2,10)
                    y = np.linspace(-prop[key][o]["height"]/2,prop[key][o]["height"]/2,10)

                    X,Y = np.meshgrid(x,y)
                    Z = (d - a*X - b*Y) / c

                    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
                    #print('positions: ', positions)
                    ROT1 = get_rotation_matrix_bti([0, 0, prop[key][o]["rot_angle"]])
                    pos_rot = np.dot(ROT1,positions)
                    center_point = np.asarray([prop[key][o]["center2d"][0], prop[key][o]["center2d"][1], 0])
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
                    

                    # print('X', X[:].shape)
                    # print('Y', Y[:].shape)
                    # print('Z', Z[:].shape)
                    if(np.abs(eq[2]) > 0.8):
                        alp = 0.3
                    else:
                        alp = 1
                        ax7.contour(Yrot,Xrot, Zrot, 20, cmap='RdGy', linewidths=2)
                    
                    surf = ax4.plot_surface(Xrot, Yrot, Zrot, alpha=alp)

                    # ax4.plot(xx_p, xy_p , label='Posição')

                    
                break
    ax7.grid()
    ax7.axis('equal')
    ax7.plot(xy_p, xx_p, "--")
    ax7.scatter(xy_p[-1], xx_p[-1] , marker='o')

    ax4.legend(loc="upper right")
    ax4.grid()
    xz_p =  np.stack([0]*xy_p.shape[0],0)
    ax4.plot(xx_p, xy_p,xz_p, "--")
    ax4.scatter(xx_p[-1], xy_p[-1],xz_p[-1], marker='o')
    ax4.view_init(-140, -30)


    axisEqual3D(ax4)
    
    #print("ULTIMO P:\n",P_p_list_total[-1])
    ax5.matshow(P_p_list_total[-1])
    ax8.matshow(x_p_list_total[-1].T)
    for (i, j), z in np.ndenumerate(x_p_list_total[-1].T):
        ax8.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    totap = []
    for p_p in P_p_list:
        # print('p_p', p_p)
        totap.append(np.sqrt(np.linalg.det(p_p)))

    ax6.plot(totap, label='Incerteza')
    ax6.legend(loc="upper right")
    ax6.grid()


    plt.show()
    # print("xp: ", x_p_list)
    # print("pp: ", P_p_list)


while True:

    f = open('ekf.pckl', 'rb')
    ekflist = pickle.load(f)
    f.close()

    f = open('feat.pckl', 'rb')
    obj = pickle.load(f)
    f.close()

    threaded_function(ekflist, obj)

    time.sleep(0.1)
    


