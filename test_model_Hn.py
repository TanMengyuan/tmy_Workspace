"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc:
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOM_SIZE = np.array([10, 10])
DNA_SIZE = ROOM_SIZE[0] * ROOM_SIZE[1]  # DNA length
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
ngx, ngy = dimX * 10, dimY * 10
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
nLed = 60
x = np.linspace(0 + 0.05, dimX - 0.05, ngx)
y = np.linspace(0 + 0.05, dimY - 0.05, ngy)
xr, yr = np.meshgrid(x, y)
Hn_value_data = np.array([])
Pt = 0.02
Pt *= nLed * nLed

for i in range(ROOM_SIZE[0]):
    for j in range(ROOM_SIZE[1]):
        Hn_value_data = np.append(Hn_value_data,
                                  np.load(r'Hn_value_data/Hn_value_%s.npy' % (str(i) + str(j))))
Hn_value_data = Hn_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)


def plotting(DNA):
    DNA = DNA.reshape(-1, ROOM_SIZE[0], ROOM_SIZE[1])[0]
    xt, yt = [], []
    Pr = np.zeros((ngx, ngy))
    indexes = np.where(DNA == 1)
    led = len(indexes[0])
    for j in range(led):
        xt.append(indexes[0][j])
        yt.append(indexes[1][j])

    for k in range(len(xt)):
        Pr += Pt * Hn_value_data[xt[k]][yt[k]]
        print(np.mean(Pt * Hn_value_data[xt[k]][yt[k]]))
    print('Pr = {}'.format(np.mean(Pr)))
    print(Pr)
    Pr = 10 * np.log10(Pr * 1e3)
    print('Min: {:.1f}dBm, Max: {:.1f}dBm, Mean:{:.1f}dBm'.format(np.min(Pr), np.max(Pr), np.mean(Pr)))
    Pr[np.isinf(Pr)] = -1e1
    # tmp = np.zeros((50, 50, 3))
    # for i in range(50):
    #     for j in range(50):
    #         tmp[i][j] = [i, j, Pr[i][j]]
    # print(tmp)
    # np.save('tmp.npy', tmp)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xr, yr, Pr, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Received Power (dBm)')
    ax.set_zlim(-6, 4)
    plt.show()


dna = np.zeros((1, 100))
d = 2
nd = 9 - d
# dna[0][d * 11] = dna[0][nd * 10 + d] = dna[0][d * 10 + nd] = dna[0][nd * 11] = 1
one = [27, 77, 52]
dna[0][one] = 1
plotting(dna)
