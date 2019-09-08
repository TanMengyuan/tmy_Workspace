# 2018.12.04 123行需要解决逐个进行与运算的问题
# 2018.12.11 在有墙隔着的时候要考虑灯照射的盲区
# 2018.12.25 原始数据可能有个转置方面的问题
#
#
#

import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import tkinter.messagebox
import time

is_debugging = False
is_saving = False
show_detail = True

ROOM_SIZE = np.array([10, 10])
DNA_SIZE = ROOM_SIZE[0] * ROOM_SIZE[1]  # DNA length
THRESHOLD = 0.7
if is_debugging:
    N_GENERATIONS = 50
    POP_SIZE = 5  # population size
else:
    POP_SIZE = 200  # population size
    N_GENERATIONS = 5000
CROSS_RATE = 0.8  # mating probability (DNA crossover)
plt.figure(figsize=(7, 7))  # set the figure size

# set parameters
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

# load data
E_value_data = np.array([])
for i in range(ROOM_SIZE[0]):
    for j in range(ROOM_SIZE[1]):
        # E_value_data = np.append(E_value_data,
        #                          np.load(r'E_value_data/E_value_%s.npy' % (str(i) + str(j))))
        E_value_data = np.append(E_value_data,
                                 np.load(r'E_value_data_onetime_reflection/E_value_%s.npy' % (str(i) + str(j))))
E_value_data = E_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)

# id_num = np.load('log.npy')
id_num = 13
room_id = str(id_num).zfill(3)
room = np.load('room_data/%s.npy' % room_id)
# room = np.ones((10, 10))
#
# room[:1, :3] = 0
# room[-3:, :3] = 0

room_area = len(np.where(room == 1)[0])
led_num = np.int((room_area / 25) - 1e-3) + 1
# led_num = 4
repeat_arr = np.ones(10, dtype=np.int) * 5
room_mut = np.repeat(room, repeat_arr, axis=0)
room_mut = np.repeat(room_mut, repeat_arr, axis=1)
room_xx, room_yy = np.where(room == 0)[0] / 2 + 0.25, np.where(room == 0)[1] / 2 + 0.25


def plotting(DNA, gen, saving_pic, is_ending):
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

    ax1 = plt.subplot(212)
    ax1.cla()
    max_value_idx = np.argmax(value_container)
    show_max = 'value = %s' % str(round(value_container[max_value_idx][0], 3))
    ax1.plot(max_value_idx, value_container[max_value_idx], 'rs')
    ax1.annotate(show_max,
                 xytext=(max_value_idx * 0.8, value_container[max_value_idx]),
                 xy=(max_value_idx, value_container[max_value_idx]))
    if is_ending:
        ax1.plot(range(len(value_container[:max_value_idx + 1])), value_container[:max_value_idx + 1], 'k')
    else:
        ax1.plot(range(len(value_container)), value_container, 'k')
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('UIR Value')
    ax2 = ax1.twinx()
    if is_ending:
        ax2.plot(range(len(Emin_container[:max_value_idx + 1])), Emin_container[:max_value_idx + 1], 'r')
    else:
        ax2.plot(range(len(Emin_container)), Emin_container, 'r')
    ax2.set_ylim(0, 500)
    ax2.set_ylabel('Min Illuminance (lx)')
    plt.subplot(221)
    fig = plt.gcf()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(xr, yr, E.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Horizontal Illuminance (lx)')
    ax.set_title('Generations : %d ' % gen)
    plt.subplot(222)
    plt.scatter(x, y)
    plt.scatter(room_xx, room_yy, s=[280], marker='s', c='gray')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Generations : %d ' % gen)

    if saving_pic:
        plt.savefig('room_result/%s_fig.jpg' % room_id)
    plt.grid()
    plt.pause(0.1)


def UIR_fun(Emin, Emean): return Emin / Emean


def Q_fun(Emax, Emin): return 1 - ((Emax - Emin) / Emax)


def LED_fun(cur, tar): return np.abs(cur - tar)


def Ilu_fun(cur, tar): return np.abs(cur - tar)


def get_common(loc):
    return np.argsort(np.bincount(loc))[::-1][:led_num]


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

        led_gap = np.append(led_gap, LED_fun(cur=led, tar=led_num))
        value_orig = np.append(value_orig, UIR_fun(Emin=E_min, Emean=E_avg))
        # value_orig = np.append(value_orig, E_avg)
    value = value_orig
    value[led_gap != 0] = 0

    return value, (E_min, E_max, E_avg)  # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): return pred  # + 1e-3 - np.min(pred)


# def get_fitness(pred): return ((pred - np.min(pred)) + 1e-3) / ((np.max(pred) - np.min(pred)) + 1e-3)

def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]


def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # choose crossover points
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    return parent


def mutate(child, rate):
    for point in range(DNA_SIZE):
        # if np.random.rand() < MUTATION_RATE:
        if np.random.rand() < rate:
            child[point] = 1 if child[point] == 0 else 0
    return child


def ask_saving_data():
    return tkinter.messagebox.askokcancel('Room %s' % room_id, 'Saving this distribution?')


def run(pre_pop):
    global value_container, Emin_container
    value_container = [0]
    Emin_container = [0]
    DNA_saver = None
    MUTATION_RATE = 0.02  # mutation probability
    Flag = False

    # Start initialization
    # pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA
    if pre_pop is None:
        pop = np.array([])
        for _ in range(POP_SIZE):
            each = np.zeros((1, DNA_SIZE))
            each[0][np.random.choice(DNA_SIZE, led_num, replace=False)] = 1
            pop = np.append(pop, each)
        pop = pop.reshape(POP_SIZE, DNA_SIZE)
    else:
        pop = pre_pop

    pop_and = np.tile(room.reshape((1, DNA_SIZE)), (POP_SIZE, 1))

    plt.ion()

    for count in range(N_GENERATIONS):
        pop = pop * pop_and
        F_values, _ = F(pop)  # compute function value by extracting DNA

        # GA part (evolution)
        fitness = get_fitness(F_values)
        if count % 50 == 0:
            most_fitted_DNA = pop[np.argmax(fitness), :].reshape(1, DNA_SIZE)
            value, detail = F(most_fitted_DNA)
            if value > THRESHOLD:
                Flag = True
            if show_detail:
                if value > np.max(value_container):
                    DNA_saver = [most_fitted_DNA, count]
                    print('Generations: %d value = %f position: %s' %
                          (count, value, np.where(most_fitted_DNA.reshape(1, DNA_SIZE) == 1)[1]))
                value_container.append(value)
                Emin_container.append(detail[0])
                # plotting(most_fitted_DNA, count, saving_pic=False, is_ending=False)
            elif count % 500 == 0:
                print('Generation: %d' % count)

        if count == N_GENERATIONS - 1:
            # if value < THRESHOLD:
            #     run(pre_pop=pop)
            # np.save('log.npy', np.int(id_num + 1))
            if np.max(value_container) > THRESHOLD and DNA_saver:
                # np.save('room_result/%s_out.npy' % room_id, DNA_saver[0])
                plotting(DNA_saver[0], DNA_saver[1], saving_pic=True, is_ending=True)
            else:
                # np.save('room_result/%s_out_error.npy' % room_id, np.zeros((1, DNA_SIZE)))
                print('********************************************Failed.********************************************')
                plotting(most_fitted_DNA, N_GENERATIONS, saving_pic=True, is_ending=False)
            break

        # crossover and mutate
        pop = select(pop, fitness)  # select the parent
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child, MUTATION_RATE)
            parent[:] = child  # parent is replaced by its child

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run(pre_pop=None)
