"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc:
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.figure(figsize=(6, 6))  # set the figure size
Hn_array = np.load(r'ISI_array_data/Hn_array_%s.npy' % (str(2) + str(2)))
data_rate = 100 * 1e6 # b/s
T = 1 / data_rate # s
T_half = T / 2
num_of_dt = 2 * 100
dt = T_half / 100
t_array = np.linspace(0, 2 * T - dt, num_of_dt) * 1e9 # for plotting

array = Hn_array[1][1] / np.max(Hn_array[1][1])
plt.tick_params(direction='in')
# plt.plot(t_array[1:], array[1:], c='red')
plt.plot(t_array[:], array[:])
plt.xlabel('Propagation Delay Time /ns')
plt.ylabel('Normalized Impulse Response')
plt.xticks(np.arange(0, 25, 5))
# plt.ylim(-0.05, 1.05)
plt.show()
# for ii in range(50):
#     for jj in range(50):
#         print(Hn_array[ii][jj][20:])
#         N_ISI += Hn_array[ii][jj][int(gap[ii][jj] // dt):]

