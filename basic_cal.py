import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def set_LED_position(array, loc):
    for each in loc:
        array[each[0]][each[1]] = 1
    return array


def plotting(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Horizontal Illuminance (lx)')
    plt.show()


def UIR_fun(Emin, Emean): return Emin / Emean


def Q_fun(Emax, Emin): return 1 - (Emax - Emin) / Emax


def F(room, source):
    tetha = 45
    tethaHalf = 60
    m = np.int(- np.log(2) / np.log(np.cos(np.deg2rad(tethaHalf))))
    I0 = 0.73
    nLed = 60
    dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
    ht, hr = dimZ, REC_HEIGHT
    htr = ht - hr

    indexes = np.where(source == 1)
    # xt, yt = np.array([]), np.array([])
    # for j in range(len(indexes[0])):
    #     xt = np.append(xt, (indexes[1][j] // 10) / 2)
    #     yt = np.append(yt, (indexes[1][j] % 10) / 2)
    amp = 0.5
    amp_m = 1 - amp
    xt, yt = [5 * amp, 5 * amp, 5 * amp_m, 5 * amp_m], [5 * amp, 5 * amp_m, 5 * amp, 5 * amp_m]
    ngx, ngy = dimX * 10, dimY * 10
    x = np.linspace(0, dimX, ngx)
    y = np.linspace(0, dimY, ngy)
    # print(xt, yt)
    xr, yr = np.meshgrid(x, y)
    E = np.zeros((ngx, ngy))

    for i in range(len(xt)):
        d = np.sqrt(np.square(xr - xt[i]) + np.square(yr - yt[i]) + np.square(htr))
        cosTetha = htr / d
        E += (I0 * cosTetha * np.cos(np.deg2rad(tetha)) ** m) / np.square(d)

    E = E * nLed * nLed
    E = E * room

    # plotting(xr, yr, E)

    E[E == 0] = np.nan
    res_min, res_max, res_avg, res_var = np.nanmin(E), np.nanmax(E), np.nanmean(E), np.nanvar(E)
    print('Min = %.1f, Max = %.1f, Avg = %.1f' % (np.nanmin(E), np.nanmax(E), np.nanmean(E) / 4))
    # print('Var = %.1f' % np.nanvar(E))
    # print('sum = %d ' % (np.nansum(E) // 1e4))
    # print('peer_sum = %d ' % (np.nansum(E) / (50 * 50)))

    UIR = UIR_fun(Emin=res_min, Emean=res_avg)
    Q_value = Q_fun(Emax=res_max, Emin=res_min)

    return UIR, Q_value


if __name__ == '__main__':
    demo_room = np.ones((50, 50))
    # demo_room[50 // 2:, :] = 1
    # demo_room[:, 50 // 2:] = 1
    demo_source = np.zeros((1, 100))
    demo_source[0][0] = demo_source[0][1] = demo_source[0][11] = demo_source[0][10] = 1
    UIR, Q = F(demo_room, demo_source)
    print('%.3f, %.3f' % (UIR, Q))
