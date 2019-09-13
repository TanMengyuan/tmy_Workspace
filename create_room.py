import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

is_saving = True
plt.figure(figsize=(6, 6))  # set the figure size


def plot_loc(room, room_id):
    plt.cla()
    room_xx, room_yy = np.where(room == 0)[0] / 2 + 0.25, np.where(room == 0)[1] / 2 + 0.25
    plt.scatter(room_xx, room_yy, s=[1200], marker='s', c='gray')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    # plt.title("Room id : %s" % room_id)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    if is_saving:
        plt.savefig("room_shape_fig/%s.jpg" % room_id)
    # plt.pause(0.5)


def process_cut(cut):
    # 5 function to cut the room
    res = np.array([])
    room = np.ones((10, 10))
    res = np.append(res, room * cut)
    res = np.append(res, room * cut * np.rot90(cut))
    res = np.append(res, room * cut * np.rot90(cut, k=2))
    res = np.append(res, room * cut * np.rot90(cut) * np.rot90(cut, k=2))
    res = np.append(res, room * cut * np.rot90(cut) * np.rot90(cut, k=2) * np.rot90(cut, k=3))

    return res.reshape(-1, 10, 10)


rooms = np.array([])
cut = np.array([])
room_orig = np.ones((10, 10))

# square shape cut
for L in range(1, 5):
    square = np.ones((10, 10))
    cut_each = np.ones((10, 10))
    cut_each[:L, :L] = 0
    cut = np.append(cut, cut_each)

# circle shape cut
for R in range(1, 5):
    square = np.ones((10, 10))
    cut_each = np.ones((10, 10))
    for i in range(5):
        for j in range(5):
            cut_each[i][j] = 0 if np.square(i + 0.5) + np.square(j + 0.5) <= np.square(R) else 1
    cut = np.append(cut, cut_each)

# triangle shape cut
for S in range(1, 5):
    square = np.ones((10, 10))
    cut_each = np.ones((10, 10))
    for i in reversed(range(S)):
        for j in reversed(range(S - i)):
            cut_each[i][j] = 0
    cut = np.append(cut, cut_each)

plt.ion()
cut = cut.reshape(-1, 10, 10)
id_num = 0
for each in cut:
    data = process_cut(each)
    for datum in data:
        room_id = str(id_num).zfill(3)
        plot_loc(datum, room_id)
        if is_saving:
            np.save('room_data/%s.npy' % room_id, datum)
        id_num += 1
# plt.ioff(); plt.show()
