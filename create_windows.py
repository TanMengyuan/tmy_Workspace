import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

is_saving = False
plt.figure(figsize=(6, 6))  # set the figure size


def plot_loc(windows, win_id):
    plt.cla()
    room_xx, room_yy = np.where(windows == 1)[0] / 2 + 0.25, np.where(windows == 1)[1] / 2 + 0.25
    # room_xx, room_yy = np.where(windows == 1)[0] / 2, np.where(windows == 1)[1] / 2
    plt.scatter(room_xx, room_yy, s=[1200], marker='s', c='blue', alpha=0.5)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.title("Windows id : %s" % win_id)
    if is_saving:
        plt.savefig("windows_fig/%s.jpg" % win_id)
    plt.pause(0.5)


def process_win(mini_win):
    res = np.array([])
    room = np.zeros((10, 10))
    room += mini_win
    res = np.append(res, room)
    room += np.rot90(mini_win, k=2)
    res = np.append(res, room)
    room = np.zeros((10, 10))
    room += mini_win + np.rot90(mini_win)
    res = np.append(res, room)
    room += np.rot90(mini_win, k=-1)
    res = np.append(res, room)
    res[res >= 1] = 1

    return res.reshape(-1, 10, 10)


id_num = 0
for start in range(5):
    win_r = np.zeros((10, 10))
    win_r[start:10 - start, 0] = 1
    data = process_win(win_r)
    for each in data:
        room_id = str(id_num).zfill(3)
        plot_loc(each, room_id)
        if is_saving:
            np.save('windows_data/%s.npy' % room_id, each)
        id_num += 1

plt.ioff()
plt.show()
