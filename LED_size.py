import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 2.5, 0.85
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
c = 3e8
dt = 0.1
LED_length = 0.6 # m
LED_num = 60

t_array = np.arange(0, 20, dt)
t_arrive = {}
device = [0.01, 0.01]
for x in np.linspace(1.25 - LED_length / 2, 1.25 + LED_length / 2, LED_num):
    for y in np.linspace(1.25 - LED_length / 2, 1.25 + LED_length / 2, LED_num):
        t = np.sqrt(np.square(x - device[0]) + np.square(y - device[1]) + np.square(htr)) / c
        t *= 1e9
        t = round(t, 2)
        if t in t_arrive.keys():
            t_arrive[t] += 1
        else:
            t_arrive[t] = 1

t_arrive = sorted(t_arrive.items(), key=lambda i:i[0])
first_arrive = t_arrive[0][0]
start_zero = np.zeros(200)
for k, v in t_arrive:
    start_zero[int((k - first_arrive) // dt) + 1] += v
plt.plot(t_array, start_zero / np.max(start_zero))
plt.xlabel('Propagation Delay Time [ns]')
plt.ylabel('Normalized Impulse Response')
plt.show()