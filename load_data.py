import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import tkinter.messagebox
import time
from PIL import Image

id_num = 0

ROOM_SIZE = np.array([10, 10])
DNA_SIZE = ROOM_SIZE[0] * ROOM_SIZE[1]  # DNA length
tetha = 45
tethaHalf = 60
m = np.int(- np.log(2) / np.log(np.cos(np.deg2rad(tethaHalf))))
I0 = 0.73
nLed = 60
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
ngx, ngy = dimX * 10, dimY * 10
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
x = np.linspace(0 + 0.05, dimX - 0.05, ngx)
y = np.linspace(0 + 0.05, dimY - 0.05, ngy)
xr, yr = np.meshgrid(x, y)
E_value_data = np.array([])
for i in range(ROOM_SIZE[0]):
    for j in range(ROOM_SIZE[1]):
        E_value_data = np.append(E_value_data,
                                 np.load(r'E_value_data/E_value_%s.npy' % (str(i) + str(j))))
E_value_data = E_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
# id_num = np.load('log.npy')
room_id = str(id_num).zfill(3)
room = np.load('room_data/%s.npy' % room_id)
room_area = len(np.where(room == 1)[0])
led_num = round((room_area / 25), 3)
repeat_arr = np.ones(10, dtype=np.int) * 5
room_mut = np.repeat(room, repeat_arr, axis=0)
room_mut = np.repeat(room_mut, repeat_arr, axis=1)
room_xx, room_yy = np.where(room == 0)[0] / 2 + 0.25, np.where(room == 0)[1] / 2 + 0.25


def plotting(DNA, gen):
    plt.cla()
    DNA = DNA.reshape(-1, ROOM_SIZE[0], ROOM_SIZE[1])[0]
    xt, yt = [], []
    x, y = np.array([]), np.array([])
    E = np.zeros((ngx, ngy))
    indexes = np.where(DNA == 1)
    led = len(indexes[0])
    for j in range(led):
        xt.append(indexes[0][j])
        yt.append(indexes[1][j])
        x = np.append(x, indexes[0][j] / 2 + 0.25)
        y = np.append(y, indexes[1][j] / 2 + 0.25)

    for k in range(len(xt)):
        E += E_value_data[xt[k]][yt[k]]
    E = E * nLed * nLed * room_mut

    plt.subplot(121)
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xr, yr, E.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Horizontal Illuminance (lx)')
    ax.set_title('Generations : %d ' % gen)
    plt.subplot(122)
    plt.scatter(x, y)
    plt.scatter(room_xx, room_yy, s=[1250], marker='s', c='gray')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Generations : %d ' % gen)
    # if is_saving:
    #     plt.savefig('pic/generations_%d.jpg' % gen)
    # if not show_detail:
    #     plt.savefig('pic/tmp.jpg')
    # plt.pause(0.1)
    plt.show()


def run():
    DNA = np.load('room_result/%s_out.npy' % room_id)
    plotting(DNA, 2000)


if __name__ == '__main__':
    run()
