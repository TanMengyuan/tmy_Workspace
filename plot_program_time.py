"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc: Smooth the data for plot
"""
import numpy as np
import matplotlib.pyplot as plt


def smooth(arr, weight=0.5):
    last = arr[0]
    smoothed = []
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


time_saver_GA = np.load("time_saver.npy")
time_saver_DL = np.load("time_saver_DL.npy")
# time_saver_DL = np.load("time_saver_DL_part.npy")
time_saver_GA *= 1e3
print(time_saver_DL)
times_GA = smooth(arr=time_saver_GA, weight=0.5)
times_DL = smooth(arr=time_saver_DL, weight=0.1)


plt.tick_params(direction='in')
plt.xlabel('Step')
plt.ylabel('Time /s')
plt.ylim(1e0, 1e5)
plt.semilogy(times_DL)
# plt.semilogy(times_GA)

# plt.plot(times)
plt.show()
