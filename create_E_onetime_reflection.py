"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc: consider the onetime reflection of E
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tetha = np.deg2rad(45)
Rho = 0.8  # Spectral reflectance of plaster wall
FOV = np.deg2rad(60)  # FOV of receiver
tethaHalf = np.deg2rad(60)
m = np.int(- np.log(2) / np.log(np.cos(tethaHalf)))
I0 = 0.73
nLed = 60
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
# dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 2.5, 0.85
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
ngx, ngy = dimX * 10, dimY * 10
x = np.linspace(0 + dimX / (2 * ngx), dimX - dimX / (2 * ngx), ngx)
y = np.linspace(0 + dimY / (2 * ngy), dimY - dimY / (2 * ngy), ngy)
xr, yr = np.meshgrid(x, y)  # Receiver array
xt = np.linspace(0 + 0.25, 5 - 0.25, 10)  # Source array
yt = np.linspace(0 + 0.25, 5 - 0.25, 10)  # Source array
d_wall = dimX * htr / 100  # d_wall = 0.1075


# Flag = True

def plotting(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Horizontal Illuminance (lx)')
    ax.set_zlim(-300, 1300)
    plt.show()


for i in range(len(xt)):
    for j in range(len(yt)):
        d = np.sqrt(np.square(xr - xt[i]) + np.square(yr - yt[j]) + np.square(htr))
        cosTetha = htr / d
        E = (I0 * cosTetha * np.cos(tetha) ** m) / np.square(d)
        adding = np.zeros((50, 50))

        for x in [0 + 0.25, 5 - 0.25]:
            for y in np.linspace(0 + 0.25, 5 - 0.25, 10):
                for z in np.linspace(REC_HEIGHT + 0.1075, dimZ - 0.1075, 10):
                    d1 = np.sqrt(np.square(x - xt[i]) + np.square(y - yt[j]) + np.square(z - dimZ))
                    d2 = np.sqrt(np.square(xr - x) + np.square(yr - y) + np.square(z))
                    gamma1 = np.arcsin((dimZ - z) / d1)
                    gamma2 = np.arcsin(z / d2)
                    tetha_R = (np.pi / 2) - gamma2
                    tmp = ((Rho * I0 * d_wall * np.cos(tetha) ** m * np.cos(gamma1) * np.cos(gamma2) * np.cos(
                        tetha_R)) /
                           (2 * (np.pi ** 2) * d1 ** 2 * d2 ** 2))
                    # tmp = ((Rho * I0 * d_wall * np.cos((np.pi / 2) - gamma1) ** m * np.cos(gamma1) * np.cos(gamma2)
                    # * np.cos(tetha_R)) / (2 * (np.pi) * d1 ** 2 * d2 ** 2))
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
                    tmp = ((Rho * I0 * d_wall * np.cos(tetha) ** m * np.cos(gamma1) * np.cos(gamma2) * np.cos(
                        tetha_R)) /
                           (2 * (np.pi ** 2) * d1 ** 2 * d2 ** 2))
                    # tmp = ((Rho * I0 * d_wall * np.cos((np.pi / 2) - gamma1) ** m * np.cos(gamma1) * np.cos(
                    #     gamma2) * np.cos(tetha_R)) /
                    #        (2 * (np.pi) * d1 ** 2 * d2 ** 2))
                    tmp[tetha_R > FOV] = 0
                    adding += tmp

        # check data at (2, 2)
        if i == 2 and j == 2:
            print(adding)
            print(np.mean(E))
            print(np.mean(adding))
            print('{:.2f} %'.format(100 * np.mean(adding) / np.mean(E)))

        E += adding

        # np.save(r'E_value_data_onetime_reflection/E_value_%s.npy' % (str(i) + str(j)) , E.T)

    print(i, 'finish!')

print('Finish.')
