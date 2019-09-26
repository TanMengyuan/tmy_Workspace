"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc:
"""
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
MIN_EFF_AREA = 0.88
if is_debugging:
    N_GENERATIONS = 50
    POP_SIZE = 5  # population size
else:
    POP_SIZE = 100  # population size
    N_GENERATIONS = 3000
CROSS_RATE = 0.8  # mating probability (DNA crossover)
plt.figure(figsize=(12, 6))  # set the figure size

# set parameters
tethaHalf = 60
m = np.int(- np.log(2) / np.log(np.cos(np.deg2rad(tethaHalf))))
I0 = 0.73
nLed = 60
Pt = 0.02  # W
Pt *= nLed * nLed
gamma = 0.53  # A/W
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
ngx, ngy = dimX * 10, dimY * 10
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
x = np.linspace(0 + 0.05, dimX - 0.05, ngx)
y = np.linspace(0 + 0.05, dimY - 0.05, ngy)
xr, yr = np.meshgrid(x, y)

# load data
noise_value_data, Hn_value_data = np.array([]), np.array([])

for i in range(ROOM_SIZE[0]):
    for j in range(ROOM_SIZE[1]):
        noise_value_data = np.append(noise_value_data,
                                     np.load(r'noise_value_data/noise_value_%s.npy' % (str(i) + str(j))))
        Hn_value_data = np.append(Hn_value_data,
                                  np.load(r'Hn_value_data/Hn_value_%s.npy' % (str(i) + str(j))))
noise_value_data = noise_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)
Hn_value_data = Hn_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)

id_num = np.load('log.npy')
# id_num = 0
room_id = str(id_num).zfill(3)
room = np.load('room_data/%s.npy' % room_id)
# room = np.ones((10, 10))
#
# room[:1, :3] = 0
# room[-3:, :3] = 0

room_area = len(np.where(room == 1)[0])
led_num = np.int((room_area / 25) - 1e-3) + 1
repeat_arr = np.ones(10, dtype=np.int) * 5
room_mut = np.repeat(room, repeat_arr, axis=0)
room_mut = np.repeat(room_mut, repeat_arr, axis=1)
room_xx, room_yy = np.where(room == 0)[0] / 2 + 0.25, np.where(room == 0)[1] / 2 + 0.25


def plotting(DNA, gen, saving_pic, is_ending):
    plt.cla()
    DNA = DNA.reshape(-1, ROOM_SIZE[0], ROOM_SIZE[1])[0]
    xt, yt = [], []
    x, y = np.array([]), np.array([])
    S, N = np.zeros((ngx, ngy)) + 1e-9, np.zeros((ngx, ngy)) + 1e-9
    indexes = np.where(DNA == 1)
    led = len(indexes[0])
    for j in range(led):
        xt.append(indexes[0][j])
        yt.append(indexes[1][j])
        x = np.append(x, indexes[0][j] / 2 + 0.25)
        y = np.append(y, indexes[1][j] / 2 + 0.25)

    for k in range(len(xt)):
        S += (gamma ** 2) * ((Pt * Hn_value_data[xt[k]][yt[k]]) ** 2)
        N += noise_value_data[xt[k]][yt[k]]
    SNR = 10 * np.log10(S / N) * room_mut
    effect_zone = cal_effect_zone(SNR)

    # ax1 = plt.subplot(212)
    # ax1.cla()
    # max_value_idx = np.argmax(value_container)
    # show_max = 'value = %s' % str(round(value_container[max_value_idx][0], 3))
    # ax1.plot(max_value_idx, value_container[max_value_idx], 'rs')
    # ax1.annotate(show_max,
    #              xytext=(max_value_idx * 0.8, value_container[max_value_idx]),
    #              xy=(max_value_idx, value_container[max_value_idx]))
    # if is_ending:
    #     ax1.plot(range(len(value_container[:max_value_idx + 1])), value_container[:max_value_idx + 1], 'k')
    # else:
    #     ax1.plot(range(len(value_container)), value_container, 'k')
    # ax1.set_xlabel('Generations')
    # ax1.set_ylabel('Effect Area (%)')

    # plt.subplot(221)
    # fig = plt.gcf()
    # ax = fig.add_subplot(2, 2, 1, projection='3d')
    # ax.plot_surface(xr, yr, SNR.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('SNR (dB)')
    # ax.set_title('Generations : %d ' % gen)

    plt.subplot(121)
    plt.cla()
    levels = np.hstack((np.linspace(np.min(SNR[SNR != 0]), 13.6 - (np.max(SNR) - 13.6) / 4, 4),
                        np.linspace(13.6, np.max(SNR), 5))) if np.max(SNR) > 13.6 else np.linspace(0, np.max(SNR), 9)
    plt.contourf(xr, yr, SNR.T, levels=levels, alpha=.75)
    C = plt.contour(xr, yr, SNR.T, levels=levels, colors='black', linewidths=1)
    plt.clabel(C, fmt='%.1f', inline=True, fontsize=8)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('SNR (dB) Effect Area: {0} %'.format(round(round(effect_zone, 4) * 100, 2)))

    plt.subplot(122)
    plt.scatter(x, y)
    plt.scatter(room_xx, room_yy, s=[1200], marker='s', c='gray')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    # plt.title('Generations : %d ' % gen)
    plt.title('room model')

    # if saving_pic:
    # plt.savefig('room_result_SNR_contourline/%s_fig.jpg' % room_id)
    plt.grid()
    plt.pause(0.1)


def cal_effect_zone(arr): return len(arr[arr >= 13.6]) / (room_area * 25)


def LED_fun(cur, tar): return np.abs(cur - tar)


def get_common(loc):
    return np.argsort(np.bincount(loc))[::-1][:led_num]


def F(source):  # source shape [-1, 100]
    source = source.reshape(-1, ROOM_SIZE[0], ROOM_SIZE[1])
    value, value_orig, led_gap = np.array([]), np.array([]), np.array([])
    for i in range(source.shape[0]):
        indexes = np.where(source[i] == 1)
        xt, yt = [], []
        led = len(indexes[0])
        for j in range(led):
            xt.append(indexes[0][j])
            yt.append(indexes[1][j])
        S, N = np.zeros((ngx, ngy)) + 1e-9, np.zeros((ngx, ngy)) + 1e-9

        for k in range(len(xt)):
            S += (gamma ** 2) * ((Pt * Hn_value_data[xt[k]][yt[k]]) ** 2)
            N += noise_value_data[xt[k]][yt[k]]

        SNR = 10 * np.log10(S / N) * room_mut
        try:
            SNR_min, SNR_max, SNR_avg = np.min(SNR[SNR != 0]), np.max(SNR[SNR != 0]), np.mean(SNR[SNR != 0])
        except:
            SNR_min, SNR_max, SNR_avg = 1e-9, 1e-9, 1e-9

        led_gap = np.append(led_gap, LED_fun(cur=led, tar=led_num))
        value_orig = np.append(value_orig, cal_effect_zone(SNR))
    value = value_orig
    value[led_gap != 0] = 0

    return value, (SNR_min, SNR_max, SNR_avg)  # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): return pred + 1e-9  # + 1e-3 - np.min(pred)


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
        if np.random.rand() < rate:
            child[point] = 1 if child[point] == 0 else 0
    return child


def run(pre_pop):
    start_time = time.time()
    global value_container
    value_container = [0]
    DNA_saver = None
    MUTATION_RATE = 0.02  # mutation probability
    Flag = False

    # Start initialization
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
        F_values, _ = F(pop)

        # GA part (evolution)
        fitness = get_fitness(F_values)
        if count % 50 == 0:
            most_fitted_DNA = pop[np.argmax(fitness), :].reshape(1, DNA_SIZE)
            value, detail = F(most_fitted_DNA)
            if show_detail:
                if value > np.max(value_container):
                    DNA_saver = [most_fitted_DNA, count]
                    # print('Generations: %d value = %f position: %s' %
                    #       (count, value, np.where(most_fitted_DNA.reshape(1, DNA_SIZE) == 1)[1]))
                value_container.append(value)
                # plotting(most_fitted_DNA, count, saving_pic=False, is_ending=False)
            elif count % 500 == 0:
                print('Generation: %d' % count)
            if np.max(value_container) > MIN_EFF_AREA:
                # np.save('log.npy', np.int(id_num + 1))
                break

        if count == N_GENERATIONS - 1:
            # np.save('log.npy', np.int(id_num + 1))
            if DNA_saver and np.max(value_container) > MIN_EFF_AREA:
                # np.save('room_result_SNR_contourline/%s_out.npy' % room_id, DNA_saver[0])
                # plotting(DNA_saver[0], DNA_saver[1], saving_pic=True, is_ending=True)
                print('{0} Finish.'.format(room_id))
            else:
                # np.save('room_result_SNR_contourline/%s_out_error.npy' % room_id, np.zeros((1, DNA_SIZE)))
                print('********************************************Failed.********************************************')
                # plotting(most_fitted_DNA, N_GENERATIONS, saving_pic=True, is_ending=False)
            break

        # crossover and mutate
        pop = select(pop, fitness)  # select the parent
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child, MUTATION_RATE)
            parent[:] = child  # parent is replaced by its child

    end_time = time.time()
    use_time = round(end_time - start_time, 3)
    print("Program running time: {}s".format(use_time))
    time_saver = np.load("time_saver.npy")
    time_saver = np.append(time_saver, use_time)
    np.save("time_saver.npy", time_saver)
    # plt.ioff()
    # plt.show()


if __name__ == '__main__':
    run(pre_pop=None)
