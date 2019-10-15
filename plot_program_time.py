"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc: Smooth the data for plot
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def smooth(arr, weight=0.5):
    last = arr[0]
    smoothed = []
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


time_saver = np.load("time_saver.npy")
times = smooth(time_saver)
plt.tick_params(direction='in')
plt.xlabel('Step')
plt.ylabel('Time /s')
plt.plot(times)
plt.show()
