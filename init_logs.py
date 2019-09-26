"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc: initialize the counter
"""
import numpy as np

np.save('log.npy', 0)
a = np.load('log.npy')
print(a)

time_saver = np.array([])
np.save("time_saver.npy", time_saver)
print(time_saver)