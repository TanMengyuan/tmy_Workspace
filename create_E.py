import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tetha = 45
tethaHalf = 60
m = np.int(- np.log(2) / np.log(np.cos(np.deg2rad(tethaHalf))))
I0 = 0.73
nLed = 60
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
# dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 2.5, 0.85
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
ngx, ngy = dimX * 10, dimY * 10
x = np.linspace(0 + dimX / (2 * ngx), dimX - dimX / (2 * ngx), ngx)
y = np.linspace(0 + dimY / (2 * ngy), dimY - dimY / (2 * ngy), ngy)
xr, yr = np.meshgrid(x, y)
# E = np.zeros((ngx, ngy))
xt = np.linspace(0 + 0.25, 5 - 0.25, 10)
yt = np.linspace(0 + 0.25, 5 - 0.25, 10)
# print('xr\n', xr, '\nyr\n', yr)
# print('xt\n', xt, '\nyt\n', yt)
count = 0


def plotting(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Horizontal Illuminance (lx)')
    ax.set_zlim(-300, 1300)
    plt.show()


for i in range(len(xt)):
    for j in range(len(yt)):
        count += 1
        d = np.sqrt(np.square(xr - xt[i]) + np.square(yr - yt[j]) + np.square(htr))
        cosTetha = htr / d
        E = (I0 * cosTetha * np.cos(np.deg2rad(tetha)) ** m) / np.square(d)
        np.save(r'E_value_data/E_value_%s.npy' % (str(i) + str(j)), E.T)

print('Finish.')
