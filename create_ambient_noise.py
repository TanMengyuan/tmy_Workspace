"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc: create the ambient noise data at /ambient_noise_value_data
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOM_SIZE = np.array([10, 10])
DNA_SIZE = ROOM_SIZE[0] * ROOM_SIZE[1]            # DNA length
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 2, 0.85    # 窗户高度设为2m
ngx, ngy = dimX * 10, dimY * 10
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
x = np.linspace(0 + dimX / (2 * ngx), dimX - dimX / (2 * ngx), ngx)
y = np.linspace(0 + dimY / (2 * ngy), dimY - dimY / (2 * ngy), ngy)
xr, yr = np.meshgrid(x, y)
xt = np.linspace(0 + 0.25, 5 - 0.25, 10)
yt = np.linspace(0 + 0.25, 5 - 0.25, 10)

N_q = 1e4 # photons/bit
q = 1.602e-19 # C or A*s or F*V
data_rate = 100 * 1e6 # b/s
noise_bandwidth = 2 * data_rate # Hz
B = noise_bandwidth
Ibg = 5100 * 1e-6 # A
I2 = 0.562
hp = 6.62607015 * 1e-34 # J * s
c = 3e8 # m / s
lambda_w = 430 * 1e-9 # m
k = 1.38064852 * 1e-23 # J / K
TB = 6000 # K

# def W_approx(lambda_w_, TB_):
#     return ((2 * np.pi * hp * (c ** 2)) / (lambda_w_ ** 5)) / (np.e ** ((hp * c) / (lambda_w_ * k * TB_)) - 1)
#
# step_len = 200
# # xx = np.linspace(1e-9, 3000 * 1e-9, 200)
# xx = np.linspace(400 * 1e-9, 760 * 1e-9, step_len)
# yy = W_approx(xx, TB)
# print(np.nanmax(yy) / 1e13)
# dx = (760 * 1e-9 - 400 * 1e-9) / step_len
# w_a = np.sum(yy) * dx
# print(w_a / 1e7)
# plt.plot(xx, yy)
# plt.vlines(400 * 1e-9, np.nanmin(yy), np.nanmax(yy), colors = "c", linestyles = "dashed")
# plt.vlines(760 * 1e-9, np.nanmin(yy), np.nanmax(yy), colors = "c", linestyles = "dashed")
# plt.show()


for i in [0, len(xt) - 1]:
    for j in range(len(yt)):
        d = np.sqrt(np.square(xr - xt[i]) + np.square(yr - yt[j]) + np.square(htr))
        Ibg_reduce = (1 / np.square(d)) * Ibg
        ambient_noise = N_q * (2 * q * Ibg_reduce * I2 * B)
        np.save(r'./ambient_noise_value_data/ambient_noise_value_%s.npy' % (str(i) + str(j)), ambient_noise.T)

for i in range(len(xt)):
    for j in [0, len(yt) - 1]:
        d = np.sqrt(np.square(xr - xt[i]) + np.square(yr - yt[j]) + np.square(htr))
        Ibg_reduce = (1 / np.square(d)) * Ibg
        ambient_noise = N_q * (2 * q * Ibg_reduce * I2 * B)
        np.save(r'./ambient_noise_value_data/ambient_noise_value_%s.npy' % (str(i) + str(j)), ambient_noise.T)

print('Finish.')
