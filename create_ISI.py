import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tetha = np.deg2rad(45)
tethaHalf = np.deg2rad(60)
m = np.int(- np.log(2) / np.log(np.cos(tethaHalf)))
I0 = 0.73
FOV = np.deg2rad(70)
Rho = 0.2  # Spectral reflectance of plaster wall
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
c = 3e8  # m/s
data_rate = 100 * 1e6  # b/s

T = 1 / data_rate  # s
T_half = T / 2
num_of_dt = 2 * 100
dt = T_half / 100
t_array = np.linspace(0, T - dt, num_of_dt)  # for plotting


def mini_plot(array):
    fig = plt.gcf()
    ax = Axes3D(fig)
    ax.plot_surface(xr, yr, array.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.show()


def plot_Hn(array):
    # plt.plot(t_array, array, c='red')
    plt.plot(t_array[1:], array[1:], c='red')
    plt.show()


def plotting(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # ax.set_zlabel('DC gain')
    # ax.set_zlim(-300, 1300)
    # plt.pause(0.1)
    plt.show()


# save a numpy array [[arrive_time <- float], [Hn(0), Hn(dt), Hn(2 * dt), ... , Hn(200 * dt)] <- float, size(1, 200)]

for i in range(len(xt)):
    for j in range(len(yt)):
        Hn_array = np.zeros((50, 50, num_of_dt))
        d = np.sqrt(np.square(xr - xt[i]) + np.square(yr - yt[j]) + np.square(htr))
        first_arrive = d / c
        cosTetha = htr / d
        tetha_irr = np.arccos(cosTetha)
        # tetha_irr = (np.pi / 2) - np.arcsin(cosTetha)
        Hn = (Ar * (m + 1) * (np.cos(tetha_irr) ** m) * (np.square(n) /
                                                         (np.square(np.sin(FOV)))) * np.cos(tetha_irr)) / \
             (2 * np.pi * np.square(d))
        Hn[tetha_irr > FOV] = 0
        Hn_array[:, :, 0] = Hn
        # if i == 2 and j == 2:
        #     mini_plot(Hn_array[:, :, 0])
        #     plotting(xr, yr, first_arrive)
        #     print(Hn_array[:, :, 0])
        #     plotting(xr, yr, Hn_array[:, :, 0])

        # adding = np.zeros((50, 50))

        for x in [0 + 0.25, 5 - 0.25]:
            for y in np.linspace(0 + 0.25, 5 - 0.25, 10):
                for z in np.linspace(REC_HEIGHT + 0.1075, dimZ - 0.1075, 10):
                    d1 = np.sqrt(np.square(x - xt[i]) + np.square(y - yt[j]) + np.square(z - dimZ))
                    d2 = np.sqrt(np.square(xr - x) + np.square(yr - y) + np.square(z))
                    delta_t = ((d1 + d2) / c) - first_arrive
                    gamma1 = np.arcsin((dimZ - z) / d1)
                    gamma2 = np.arcsin(z / d2)
                    tetha_R = (np.pi / 2) - gamma2
                    tmp = (((m + 1) * Ar) / (2 * (np.pi ** 2) * (d1 ** 2) * (d2 ** 2))) * \
                          Rho * d_wall * (np.cos(tetha) ** m) * np.cos(gamma1) * np.cos(gamma2) * \
                          (np.square(n) / (np.square(np.sin(FOV)))) * np.cos(tetha_R)
                    tmp[tetha_R > FOV] = 0
                    for ii in range(50):
                        for jj in range(50):
                            if delta_t[ii, jj] < T:
                                ind = int(delta_t[ii, jj] // dt)
                                Hn_array[ii, jj, ind] += tmp[ii, jj]
                    # adding += tmp

        for y in [0 + 0.25, 5 - 0.25]:
            for x in np.linspace(0 + 0.25, 5 - 0.25, 10):
                for z in np.linspace(REC_HEIGHT + 0.1075, dimZ - 0.1075, 10):
                    d1 = np.sqrt(np.square(x - xt[i]) + np.square(y - yt[j]) + np.square(z - dimZ))
                    d2 = np.sqrt(np.square(xr - x) + np.square(yr - y) + np.square(z))
                    delta_t = ((d1 + d2) / c) - first_arrive
                    gamma1 = np.arcsin((dimZ - z) / d1)
                    gamma2 = np.arcsin(z / d2)
                    tetha_R = (np.pi / 2) - gamma2
                    tmp = (((m + 1) * Ar) / (2 * (np.pi ** 2) * (d1 ** 2) * (d2 ** 2))) * \
                          Rho * d_wall * (np.cos(tetha) ** m) * np.cos(gamma1) * np.cos(gamma2) * \
                          (np.square(n) / (np.square(np.sin(FOV)))) * np.cos(tetha_R)
                    tmp[tetha_R > FOV] = 0
                    for ii in range(50):
                        for jj in range(50):
                            if delta_t[ii, jj] < T:
                                ind = int(delta_t[ii, jj] // dt)
                                Hn_array[ii, jj, ind] += tmp[ii, jj]
                    # adding += tmp

        if i == 2 and j == 2:
            # print(Hn_array[1][1])
            print(np.sum(Hn_array[1][1][1:]))
            print(Hn_array[1][1][0])
            # plot_Hn(Hn_array[1][1])
            # raise SystemError
        #     print(adding)
        #     print(np.mean(adding))
        #     print(np.mean(Hn))
        #     print('{:.2f} %'.format(100 * np.mean(adding) / np.mean(Hn)))
        #     # plotting(xr, yr, adding)

        # Hn += adding

        np.save('./ISI_array_data/first_arrive_%s.npy' % (str(i) + str(j)), first_arrive.T)
        np.save('./ISI_array_data/Hn_array_%s.npy' % (str(i) + str(j)), Hn_array.transpose((1, 0, 2)))

    print('Finish', i)

print('Finish.')
