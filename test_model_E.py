import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

ROOM_SIZE = np.array([10, 10])
DNA_SIZE = ROOM_SIZE[0] * ROOM_SIZE[1]  # DNA length
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
        # E_value_data = np.append(E_value_data,
        #                          np.load(r'E_value_data/E_value_%s.npy' % (str(i) + str(j))))
        E_value_data = np.append(E_value_data,
                                 np.load(r'E_value_data_onetime_reflection/E_value_%s.npy' % (str(i) + str(j))))
E_value_data = E_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)

# id_num = 19
# room_id = str(id_num).zfill(3)
# room = np.load('room_data/%s.npy' % room_id)
room = np.ones((10, 10))
# room_area = len(np.where(room == 1)[0])
# led_num = round((room_area / 25), 3)
# led_num = np.int((room_area / 25) - 1e-3) + 1
led_num = 4
repeat_arr = np.ones(10, dtype=np.int) * 5
room_mut = np.repeat(room, repeat_arr, axis=0)
room_mut = np.repeat(room_mut, repeat_arr, axis=1)
room_xx, room_yy = np.where(room == 0)[0] / 2 + 0.25, np.where(room == 0)[1] / 2 + 0.25


def Q_fun(Emax, Emin): return 1 - ((Emax - Emin) / Emax)


def UIR_fun(Emin, Emean): return Emin / Emean


def LED_fun(cur, tar): return np.abs(cur - tar)


def F(source):  # source shape [-1, 100]
    source = source.reshape(-1, ROOM_SIZE[0], ROOM_SIZE[1])
    value, value_orig, Lum, led_gap = np.array([]), np.array([]), np.array([]), np.array([])
    for i in range(source.shape[0]):
        indexes = np.where(source[i] == 1)
        xt, yt = [], []
        led = len(indexes[0])
        for j in range(led):
            xt.append(indexes[0][j])
            yt.append(indexes[1][j])
        E = np.zeros((ngx, ngy))

        for k in range(len(xt)):
            E += E_value_data[xt[k]][yt[k]]

        E = E * nLed * nLed * room_mut
        try:
            E_min, E_max, E_avg = np.min(E[E != 0]), np.max(E[E != 0]), np.mean(E[E != 0])
        except:
            E_min, E_max, E_avg = 0, 1e-3, 1e-3

        # Heighter is better (between 0 to 1)
        # try:
        #     Lum = np.append(Lum, E_avg / (180 * led))
        # except:
        #     Lum = np.append(Lum, 0)
        # led_gap = np.append(led_gap, LED_fun(cur=led, tar=led_num))
        value_orig = np.append(value_orig, UIR_fun(Emin=E_min, Emean=E_avg))
    value = value_orig  # * 0.6
    # Lum_value = ((Lum - np.min(Lum)) / (np.max(Lum) - np.min(Lum)))
    # value += Lum_value * 0.1
    # value[led_gap != 0] = 0
    gap_value = Lum_value = None
    print('E_min = %d, E_max = %d, E_avg = %d' % (E_min, E_max, E_avg))
    # var = np.var(E)
    # print(var)
    # tmp = (np.max(led_gap) + 1) / (led_gap + 1)      # LED numbers gap
    # gap_value = (tmp / np.max(tmp))
    # value +=  gap_value * 0.3

    # if False in (value <= 1):
    #     print(value)
    # assert False not in (value <= 1)

    return value, value_orig, Lum_value, gap_value  # to find the maximum of this function


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
    E *= nLed * nLed * room_mut

    # # adding some test
    # E[E < 600] = 0
    # ratio = len(E[E == 0]) / len(E)
    # print(ratio)

    print('Min : {:.1f}lx, Max : {:.1f}lx, Mean : {:.1f}lx'.format(np.min(E), np.max(E), np.mean(E)))

    plt.subplot(121)
    fig = plt.gcf()
    fig.set_size_inches(12, 5.5)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xr, yr, E.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_zlim(-200, 900)
    # ax.set_zlim(-200, 1300)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Horizontal Illuminance (lx)')
    # ax.set_title('Generations : %d ' % gen)
    plt.subplot(122)
    plt.scatter(x, y)
    plt.scatter(room_xx, room_yy, s=[1250], marker='s', c='gray')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    # plt.title('Generations : %d ' % gen)
    # if not show_detail:
    #     plt.savefig('pic/tmp.jpg')
    # plt.pause(0.1)
    plt.grid()
    plt.show()


dna = np.zeros((1, 100))
d = 2
nd = 9 - d
dna[0][d * 11] = dna[0][nd * 10 + d] = dna[0][d * 10 + nd] = dna[0][nd * 11] = 1
# dna[0][22] = dna[0][77] = dna[0][27] = dna[0][63] = 1
# dna[0][22] = dna[0][72] = dna[0][57] = 1
# dna[0][00] = 1
plotting(dna, 0)
