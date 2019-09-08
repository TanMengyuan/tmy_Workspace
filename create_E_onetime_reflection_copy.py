#   2019.01.16 FOV的问题需要处理，需要排除掉接收的角度大于FOV的光线传入接收器。 Done


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tetha = np.deg2rad(45)
Rho = 0.8  # Spectral reflectance of plaster wall
FOV = np.deg2rad(60)  # FOV of receiver
tethaHalf = 60
m = np.int(- np.log(2) / np.log(np.cos(np.deg2rad(tethaHalf))))
I0 = 0.73
nLed = 60
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
ngx, ngy = dimX * 10, dimY * 10
x = np.linspace(0 + dimX / (2 * ngx), dimX - dimX / (2 * ngx), ngx)
y = np.linspace(0 + dimY / (2 * ngy), dimY - dimY / (2 * ngy), ngy)
xr, yr = np.meshgrid(x, y)  # Receiver array
xt = np.linspace(0 + 0.25, 5 - 0.25, 10)  # Source array
yt = np.linspace(0 + 0.25, 5 - 0.25, 10)  # Source array
d_wall = dimX * dimZ / ngx  # d_wall = 0.3
count = 0


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
        count += 1
        d = np.sqrt(np.square(xr - xt[i]) + np.square(yr - yt[j]) + np.square(htr))
        cosTetha = htr / d
        E = (I0 * cosTetha * np.cos(tetha) ** m) / np.square(d)
        # if count == 2:
        #     print(E)

        for x in [0, 5]:
            for y in np.linspace(0 + 0.25, 5 - 0.25, 10):
                for z in np.linspace(0 + 0.25, 5 - 0.25, 10):
                    d1 = np.sqrt(np.square(x - xt[i]) + np.square(y - yt[j]) + np.square(z - dimZ))
                    d2 = np.sqrt(np.square(xr - x) + np.square(yr - y) + np.square(z))
                    gamma1 = np.arcsin((dimZ - z) / d1)
                    gamma2 = np.arcsin(z / d2)
                    tetha_R = (np.pi / 2) - gamma2
                    tetha_R[tetha_R < FOV] = 0
                    E += ((Rho * I0 * d_wall * np.cos(tetha) ** m * np.cos(gamma1) * np.cos(gamma2) * np.cos(tetha_R)) /
                          (np.pi * d1 ** 2 * d2 ** 2)) * tetha_R

        for y in [0, 5]:
            for x in np.linspace(0 + 0.25, 5 - 0.25, 10):
                for z in np.linspace(0 + 0.25, 5 - 0.25, 10):
                    d1 = np.sqrt(np.square(x - xt[i]) + np.square(y - yt[j]) + np.square(z - dimZ))
                    d2 = np.sqrt(np.square(xr - x) + np.square(yr - y) + np.square(z))
                    gamma1 = np.arcsin((dimZ - z) / d1)
                    gamma2 = np.arcsin(z / d2)
                    tetha_R = (np.pi / 2) - gamma2
                    tetha_R[tetha_R < FOV] = 0
                    E += ((Rho * I0 * d_wall * np.cos(tetha) ** m * np.cos(gamma1) * np.cos(gamma2) * np.cos(tetha_R)) /
                          (np.pi * d1 ** 2 * d2 ** 2)) * tetha_R

        # if count == 2:
        #     plotting(xr, yr, E * 3600)

        np.save(r'E_value_data_onetime_reflection/E_value_%s.npy' % (str(i) + str(j)), E.T)

print('Finish.')
