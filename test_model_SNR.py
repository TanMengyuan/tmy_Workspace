"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc: Show the SNR
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy
import scipy.signal

hparams_justify = 1
# hparams_justify = 4 / 3

plt.figure(figsize=(12, 6))  # set the figure size
# plt.figure(figsize=(12, 6))  # set the figure size
ROOM_SIZE = np.array([10, 10])
DNA_SIZE = ROOM_SIZE[0] * ROOM_SIZE[1]  # DNA length
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
ngx, ngy = dimX * 10, dimY * 10
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
x = np.linspace(0 + dimX / (2 * ngx), dimX - dimX / (2 * ngx), ngx)
y = np.linspace(0 + dimY / (2 * ngy), dimY - dimY / (2 * ngy), ngy)
xr, yr = np.meshgrid(x, y)
xt = np.linspace(0 + 0.25, 5 - 0.25, 10)
yt = np.linspace(0 + 0.25, 5 - 0.25, 10)

c = 3e8  # m/s
nLed = 60
# nLed = 60
Pt = 0.02  # W
Pt *= nLed * nLed
gamma = 0.53  # A/W
# data_rate = 2000 * 1e6 # b/s
data_rate = 100 * 1e6  # b/s
T = 1 / data_rate  # s
T_half = T / 2
num_of_dt = 2 * 100
dt = T_half / 100
t_array = np.linspace(0, T - dt, num_of_dt)  # for plotting

noise_value_data, first_arrive_data, Hn_value_data = np.array([]), np.array([]), np.array([])
ambient_noise_value_data, n_shot_noise_value_data, n_thermal_noise_value_data = np.array([]), np.array([]), np.array([])
E_value_data = np.array([])

for i in range(ROOM_SIZE[0]):
    for j in range(ROOM_SIZE[1]):
        E_value_data = np.append(E_value_data,
                                 np.load(r'./E_value_data_onetime_reflection/E_value_%s.npy' % (str(i) + str(j))))
        noise_value_data = np.append(noise_value_data,
                                     np.load(r'./noise_value_data/noise_value_%s.npy' % (str(i) + str(j))))
        first_arrive_data = np.append(first_arrive_data,
                                      np.load(r'./ISI_array_data/first_arrive_%s.npy' % (str(i) + str(j))))
        Hn_value_data = np.append(Hn_value_data,
                                  np.load(r'./Hn_value_data/Hn_value_%s.npy' % (str(i) + str(j))))
        n_shot_noise_value_data = np.append(n_shot_noise_value_data,
                                            np.load(r'./n_shot_value_data/n_shot_value_%s.npy' % (str(i) + str(j))))
        n_thermal_noise_value_data = np.append(n_thermal_noise_value_data,
                                               np.load(
                                                   r'./n_thermal_value_data/n_thermal_value_%s.npy' % (str(i) + str(j))))
        if os.path.isfile(r'./ambient_noise_value_data/ambient_noise_value_%s.npy' % (str(i) + str(j))):
            ambient_noise_value_data = np.append(ambient_noise_value_data,
                                                 np.load(r'./ambient_noise_value_data/ambient_noise_value_%s.npy' % (
                                                             str(i) + str(j))))
        else:
            ambient_noise_value_data = np.append(ambient_noise_value_data, np.zeros((50, 50)))

noise_value_data = noise_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
ambient_noise_value_data = ambient_noise_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
first_arrive_data = first_arrive_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
Hn_value_data = Hn_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
n_shot_noise_value_data = n_shot_noise_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
n_thermal_noise_value_data = n_thermal_noise_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
E_value_data = E_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)


def mini_plot(array):
    fig = plt.gcf()
    ax = Axes3D(fig)
    ax.plot_surface(xr, yr, array.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.show()


def plotting(DNA, id_num):
    room_id = str(id_num).zfill(3)
    room = np.load('room_data/%s.npy' % room_id)
    # print(np.rot90(room))
    # room = np.ones((10, 10))
    room_area = len(np.where(room == 1)[0])
    repeat_arr = np.ones(10, dtype=np.int) * 5
    room_mut = np.repeat(room, repeat_arr, axis=0)
    room_mut = np.repeat(room_mut, repeat_arr, axis=1)
    x, y = np.array([]), np.array([])
    room_xx, room_yy = np.where(room == 0)[0] / 2 + 0.25, np.where(room == 0)[1] / 2 + 0.25

    DNA = DNA.reshape(-1, ROOM_SIZE[0], ROOM_SIZE[1])[0]
    xt, yt = [], []
    S, N, E, ambient_noise, t_min = np.zeros((ngx, ngy)), np.zeros((ngx, ngy)), np.zeros((ngx, ngy)), \
                                    np.zeros((ngx, ngy)), np.ones((ngx, ngy))
    indexes = np.where(DNA == 1)
    led = len(indexes[0])
    win_xx, win_yy = [], []

    for i in range(0, 10):
        # N += ambient_noise_value_data[i][0]
        win_xx.append(i / 2 + 0.25)
        win_yy.append(-0.1)

    for j in range(led):
        xt.append(indexes[0][j])
        yt.append(indexes[1][j])
        x = np.append(x, indexes[0][j] / 2 + 0.25)
        y = np.append(y, indexes[1][j] / 2 + 0.25)

    for l in range(len(xt)):
        t_min = np.minimum(t_min, first_arrive_data[xt[l]][yt[l]])
        E += E_value_data[xt[l]][yt[l]]
    t_min *= room_mut
    # mini_plot(t_min)
    E *= nLed * nLed * room_mut
    min_E = np.min(E[E > 0])
    amp = 300 / min_E
    # hparams_justify = amp
    E *= amp

    # include_ISI = True
    include_ISI = False
    for k in range(len(xt)):
        N += n_thermal_noise_value_data[xt[k]][yt[k]] + n_shot_noise_value_data[xt[k]][yt[k]] * hparams_justify
        if include_ISI:
            N_ISI, S_ISI = np.zeros((50, 50)), np.zeros((50, 50))
            gap = t_min + T - first_arrive_data[xt[k]][yt[k]]
            Hn_array = np.load(r'ISI_array_data/Hn_array_%s.npy' % (str(xt[k]) + str(yt[k])))
            for ii in range(50):
                for jj in range(50):
                    S_ISI[ii][jj] += np.sum(Hn_array[ii][jj][:int(gap[ii][jj] // dt)])
                    N_ISI[ii][jj] += np.sum(Hn_array[ii][jj][int(gap[ii][jj] // dt):])
            S += (gamma ** 2) * ((Pt * S_ISI) ** 2)
            N += (gamma ** 2) * ((Pt * N_ISI) ** 2)
        else:
            Hn = Hn_value_data[xt[k]][yt[k]]
            Prs = Pt * Hn * hparams_justify
            S += (gamma ** 2) * (Prs ** 2)

    # mini_plot(t_min)
    # mini_plot(N)
    SNR = 10 * np.log10(S / N) * room_mut
    print('Hparams justify', hparams_justify)
    print('mean S', np.mean(S))
    print('mean N', np.mean(N))
    print('mean SNR', np.mean(SNR))
    # print('min SNR ', np.min(SNR[SNR > 0]))
    print('max SNR ', np.max(SNR), '\n')

    ratio = len(SNR[SNR > 13.6]) / (room_area * 25)
    print('Effect Area : {0}'.format(ratio))
    print('max E', np.max(E[E > 0]))
    print('min E', np.min(E[E > 0]))

    Q = np.mean(SNR) / (2 * np.var(SNR) ** 0.5)
    # mini_plot(SNR)
    print(Q)
    # mini_plot(Q)

    plt.subplot(121)
    plt.tick_params(direction='in')
    levels = np.hstack((np.linspace(np.min(SNR[SNR != 0]), 13.6 - (np.max(SNR) - 13.6) / 3, 3),
                        np.linspace(13.6 + (np.max(SNR) - 13.6) / 4, np.max(SNR), 4))) \
        if np.max(SNR) > 13.6 else np.linspace(0, np.max(SNR), 8)
    plt.contourf(xr, yr, SNR.T, levels=levels, alpha=.75)
    C = plt.contour(xr, yr, SNR.T, levels=levels, colors='black', linewidths=1)
    C_ = plt.contour(xr, yr, SNR.T, levels=[np.min(SNR[SNR != 0]), 13.6], colors='black', linewidths=3)
    # plt.clabel(C, fmt='%.1f', inline=True, fontsize=10, manual=True)
    # plt.clabel(C_, fmt='%.1f', inline=True, fontsize=10, manual=True)
    plt.clabel(C, fmt='%.1f', inline=True, fontsize=10)
    plt.clabel(C_, fmt='%.1f', inline=True, fontsize=10)
    plt.xlabel('Width /m')
    plt.ylabel('Length /m')
    plt.title('SNR /dB  Effect Area: {0} %'.format(round(round(ratio, 4) * 100, 2)))

    plt.subplot(122)
    plt.scatter(x, y)
    # plt.scatter(x, y, c='gray')
    plt.tick_params(direction='in')
    plt.scatter(room_xx, room_yy, s=[1200], marker='s', c='gray')
    # plt.scatter(win_xx, win_yy, s=[1200], marker='s', c='blue', alpha=0.6, hatch='/')
    # plt.scatter(win_xx, win_yy, s=[1077], marker='s', c='gray', alpha=0.6, hatch='/')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('Width /m')
    plt.ylabel('Length /m')
    plt.title('Room Model')

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(xr, yr, SNR.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # ax.plot_surface(xr, yr, S.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # ax.plot_surface(xr, yr, N.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('SNR (dB)')
    # ax.zaxis.get_major_formatter().set_powerlimits((0, 1))
    # ax.set_zlabel('noise')
    # ax.set_zlim(10, 26)
    plt.show()


dna = np.zeros((1, 100))
# dna = np.ones((1, 100))
d = 2
nd = 9 - d
# dna[0][d * 11] = dna[0][nd * 10 + d] = dna[0][d * 10 + nd] = dna[0][nd * 11] = 1
li = [27, 41, 75]
# li = [17, 42, 57, 82]
# li = [21, 71, 27, 77]
for each in li:
    dna[0][each] = 1
id_num = 13
plotting(dna, id_num)
