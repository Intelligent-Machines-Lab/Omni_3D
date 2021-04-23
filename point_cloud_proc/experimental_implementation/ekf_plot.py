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

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid




def threaded_function(ekflist, prop):
    x_p_list_total = ekflist['x_p_list']
    P_p_list_total = ekflist['P_p_list']
    x_p_list = []
    P_p_list = []
    for x_p in x_p_list_total:
        x_p_list.append(x_p[:3, :])
    x_p_list = np.asarray(x_p_list)

    for p_p in P_p_list_total:
        P_p_list.append(p_p[:3, :3])
    P_p_list = np.asarray(P_p_list)


    x_m_list_total = ekflist['x_m_list']
    P_m_list_total = ekflist['P_m_list']
    x_m_list = []
    P_m_list = []
    for x_m in x_m_list_total:
        x_m_list.append(x_m[:3, :])
    x_m_list = np.asarray(x_m_list)

    for p_m in P_m_list_total:
        P_m_list.append(p_m[:3, :3])
    P_m_list = np.asarray(P_m_list)




    x_real_list = np.asarray(ekflist['x_real_list'])
    x_errado_list = np.asarray(ekflist['x_errado_list'])

    print('x_real_list_total: \n',x_real_list)

    xx_p = x_p_list[:, 0, 0]
    xy_p = x_p_list[:, 1, 0]
    xtheta_p = x_p_list[:, 2, 0]

    xx_m = x_m_list[:, 0, 0]
    xy_m = x_m_list[:, 1, 0]
    xtheta_m = x_m_list[:, 2, 0]

    xx_real = x_real_list[:, 0, 0]
    xy_real = x_real_list[:, 1, 0]
    xtheta_real = x_real_list[:, 2, 0]

    xx_errado = x_errado_list[:, 0, 0]
    xy_errado = x_errado_list[:, 1, 0]
    xtheta_errado = x_errado_list[:, 2, 0]

    ppx = P_p_list[:, 0, 0]
    ppy = P_p_list[:, 1, 1]
    pptheta = P_p_list[:, 2, 2]

    pmx = P_p_list[:, 0, 0]
    pmy = P_p_list[:, 1, 1]
    pmtheta = P_p_list[:, 2, 2]

    xx_pm_list = []
    xy_pm_list = []
    xtheta_pm_list = []

    px_pm_list = []
    py_pm_list = []
    ptheta_pm_list = []

    xreal_pm_list = []
    yreal_pm_list = []
    thetareal_pm_list = []

    ipose_list = []
    for ipose in range(ppx.shape[0]):
        xx_pm_list.append(xx_p[ipose])
        xx_pm_list.append(xx_m[ipose])

        xy_pm_list.append(xy_p[ipose])
        xy_pm_list.append(xy_m[ipose])

        xtheta_pm_list.append(xtheta_p[ipose])
        xtheta_pm_list.append(xtheta_m[ipose])

        px_pm_list.append(ppx[ipose])
        px_pm_list.append(pmx[ipose])

        py_pm_list.append(ppy[ipose])
        py_pm_list.append(pmy[ipose])

        ptheta_pm_list.append(pptheta[ipose])
        ptheta_pm_list.append(pmtheta[ipose])

        xreal_pm_list.append(xx_real[ipose])
        xreal_pm_list.append(xx_real[ipose])

        yreal_pm_list.append(xy_real[ipose])
        yreal_pm_list.append(xy_real[ipose])

        thetareal_pm_list.append(xtheta_real[ipose])
        thetareal_pm_list.append(xtheta_real[ipose])

        ipose_list.append(ipose)
        ipose_list.append(ipose)


    xx_pm_list = np.asarray(xx_pm_list)
    xy_pm_list = np.asarray(xy_pm_list)
    xtheta_pm_list = np.asarray(xtheta_pm_list)

    px_pm_list = np.asarray(px_pm_list)
    py_pm_list = np.asarray(py_pm_list)
    ptheta_pm_list = np.asarray(ptheta_pm_list)

    xreal_pm_list = np.asarray(xreal_pm_list)
    yreal_pm_list = np.asarray(yreal_pm_list)
    thetareal_pm_list = np.asarray(thetareal_pm_list)

    ipose_list = np.asarray(ipose_list)

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
    print("ERROS:\n",(xreal_pm_list-xx_pm_list))

    ax1.plot(ipose_list, (xreal_pm_list-xx_pm_list), label='x_m')
    ax1.plot(ipose_list, 3*np.sqrt(px_pm_list), label='+3*sqrt(p_p_x)')
    ax1.plot(ipose_list, -3*np.sqrt(px_pm_list), label='-3*sqrt(p_p_x)')
    ax1.legend(loc="upper right")
    ax1.grid()

    ax2.plot(ipose_list, (yreal_pm_list-xy_pm_list), label='y_m')
    ax2.plot(ipose_list, 3*np.sqrt(py_pm_list), label='+3*sqrt(p_p_y)')
    ax2.plot(ipose_list, -3*np.sqrt(py_pm_list), label='-3*sqrt(p_p_y)')
    ax2.legend(loc="upper right")
    ax2.grid()

    ax3.plot(ipose_list, (thetareal_pm_list-xtheta_pm_list), label='theta_m')
    ax3.plot(ipose_list, 3*np.sqrt(ptheta_pm_list), label='+3*sqrt(p_p_theta)')
    ax3.plot(ipose_list, -3*np.sqrt(ptheta_pm_list), label='-3*sqrt(p_p_theta)')
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

            if((key == "cylinders")):
                for o in range(len(prop[key])):
                    # print("EQ. DO PLANO:",prop[key][o]["equation"])
                    center = prop[key][o]["center"]
                    radius = prop[key][o]["radius"]
                    heightvec = prop[key][o]['height']
                    height = (heightvec[1]-heightvec[0])
                    Xc,Yc,Zc = data_for_cylinder_along_z(center[0],center[1],radius,height)
                    ax4.plot_surface(Xc, Yc, Zc)
                    ax7.contour(Yc,Xc,Zc , 20, cmap='RdGy', linewidths=2)

    ax7.grid()
    ax7.axis('equal')
    ax7.plot(xy_m, xx_m, "--")
    ax7.scatter(xy_m[-1], xx_m[-1] , marker='o', label='Estimado')
    ax7.plot(xy_real, xx_real, "--")
    ax7.scatter(xy_real[-1], xx_real[-1] , marker='x', label='Real')
    ax7.plot(xy_errado, xx_errado, ".")
    ax7.scatter(xy_errado[-1], xx_errado[-1] , marker='x', label='Odom')
    ax7.legend(loc="upper right")

    ax4.legend(loc="upper right")
    ax4.grid()
    xz_m =  np.stack([0]*xy_m.shape[0],0)
    ax4.plot( xx_m,xy_m,xz_m, "--")
    ax4.scatter( xx_m[-1],xy_m[-1],xz_m[-1], marker='o', label='Estimado')
    ax4.plot( xx_real,xy_real,xz_m, "--")
    ax4.scatter( xx_real[-1],xy_real[-1],xz_m[-1] , marker='x', label='Real')
    ax4.legend(loc="upper right")
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

    class EventHandler:
        def __init__(self):
            fig.canvas.mpl_connect('button_press_event', self.onpress)

        def onpress(self, event):
            if event.inaxes!=ax8:
                return
            xi, yi = (int(round(n)) for n in (event.xdata, event.ydata))
            print(xi,yi)
            x_list = []
            y_list = []
            for n_pmat in range(len(P_p_list_total)):
                try:
                    y_list.append(3*np.sqrt(P_p_list_total[n_pmat][xi,xi]))
                    x_list.append(n_pmat)
                    print(x_list)
                    print(y_list)
                except:
                    print("não tem")
            ax6.plot(x_list, y_list, label='3xstd v '+str(n_pmat))
            ax6.grid(True)
            plt.show()
                #print(n_pmat,"   :::    ", P_p_list_total[n_pmat])


    handler = EventHandler()


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
    


