#   2019.02.28 计算Hn时用的辐射角的值等于倾斜角
#              LED高度2.5m

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tetha = np.deg2rad(45)
tethaHalf = np.deg2rad(60)
m = np.int(- np.log(2) / np.log(np.cos(tethaHalf)))
I0 = 0.73
FOV = np.deg2rad(60)
Rho = 0.8  # Spectral reflectance of plaster wall
Ar = 1e-4
n = 1.5
nLed = 60
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
# dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 2.5, 0.85
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
ngx, ngy = dimX * 10, dimY * 10
x = np.linspace(0 + dimX / (2 * ngx), dimX - dimX / (2 * ngx), ngx)
y = np.linspace(0 + dimY / (2 * ngy), dimY - dimY / (2 * ngy), ngy)
xr, yr = np.meshgrid(x, y)
xt = np.linspace(0 + 0.25, 5 - 0.25, 10)
yt = np.linspace(0 + 0.25, 5 - 0.25, 10)
d_wall = dimX * htr / 100  # d_wall = 0.1075


def plotting(x, y, z):
    plt.gcf()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('DC gain')
    # ax.set_zlim(-300, 1300)
    plt.pause(0.1)
    plt.show()


for i in range(len(xt)):
    for j in range(len(yt)):
        d = np.sqrt(np.square(xr - xt[i]) + np.square(yr - yt[j]) + np.square(htr))
        cosTetha = htr / d
        tetha_irr = np.arccos(cosTetha)
        # tetha_irr = (np.pi / 2) - np.arcsin(cosTetha)
        Hn = (Ar * (m + 1) * (np.cos(tetha_irr) ** m) * (np.square(n) /
                                                         (np.square(np.sin(FOV)))) * np.cos(tetha_irr)) / \
             (2 * np.pi * np.square(d))
        Hn[tetha_irr > FOV] = 0
        adding = np.zeros((50, 50))

        for x in [0 + 0.25, 5 - 0.25]:
            for y in np.linspace(0 + 0.25, 5 - 0.25, 10):
                for z in np.linspace(REC_HEIGHT + 0.1075, dimZ - 0.1075, 10):
                    d1 = np.sqrt(np.square(x - xt[i]) + np.square(y - yt[j]) + np.square(z - dimZ))
                    d2 = np.sqrt(np.square(xr - x) + np.square(yr - y) + np.square(z))
                    gamma1 = np.arcsin((dimZ - z) / d1)
                    gamma2 = np.arcsin(z / d2)
                    tetha_R = (np.pi / 2) - gamma2
                    tmp = (((m + 1) * Ar) / (2 * (np.pi ** 2) * (d1 ** 2) * (d2 ** 2))) * \
                          Rho * d_wall * (np.cos(tetha) ** m) * np.cos(gamma1) * np.cos(gamma2) * \
                          (np.square(n) / (np.square(np.sin(FOV)))) * np.cos(tetha_R)
                    tmp[tetha_R > FOV] = 0
                    adding += tmp

        for y in [0 + 0.25, 5 - 0.25]:
            for x in np.linspace(0 + 0.25, 5 - 0.25, 10):
                for z in np.linspace(REC_HEIGHT + 0.1075, dimZ - 0.1075, 10):
                    d1 = np.sqrt(np.square(x - xt[i]) + np.square(y - yt[j]) + np.square(z - dimZ))
                    d2 = np.sqrt(np.square(xr - x) + np.square(yr - y) + np.square(z))
                    gamma1 = np.arcsin((dimZ - z) / d1)
                    gamma2 = np.arcsin(z / d2)
                    tetha_R = (np.pi / 2) - gamma2
                    tmp = (((m + 1) * Ar) / (2 * (np.pi ** 2) * (d1 ** 2) * (d2 ** 2))) * \
                          Rho * d_wall * (np.cos(tetha) ** m) * np.cos(gamma1) * np.cos(gamma2) * \
                          (np.square(n) / (np.square(np.sin(FOV)))) * np.cos(tetha_R)
                    tmp[tetha_R > FOV] = 0
                    adding += tmp

        # Testing part
        # if i == 2 and j == 2:
        #     print(adding)
        #     print(np.mean(adding))
        #     print(np.mean(Hn))
        #     print('{:.2f} %'.format(100 * np.mean(adding) / np.mean(Hn)))
        #     plotting(xr, yr, adding)
        #
        # Hn += adding

        # np.save(r'Hn_value_data/Hn_value_%s.npy' % (str(i) + str(j)), Hn.T)

    print('Finish', i)

print('Finish.')
