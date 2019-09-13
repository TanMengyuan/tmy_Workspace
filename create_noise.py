import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOM_SIZE = np.array([10, 10])
DNA_SIZE = ROOM_SIZE[0] * ROOM_SIZE[1]  # DNA length
dimX, dimY, dimZ, REC_HEIGHT = 5, 5, 3, 0.85
ngx, ngy = dimX * 10, dimY * 10
ht, hr = dimZ, REC_HEIGHT
htr = ht - hr
x = np.linspace(0 + dimX / (2 * ngx), dimX - dimX / (2 * ngx), ngx)
y = np.linspace(0 + dimY / (2 * ngy), dimY - dimY / (2 * ngy), ngy)
xr, yr = np.meshgrid(x, y)
xt = np.linspace(0 + 0.25, 5 - 0.25, 10)
yt = np.linspace(0 + 0.25, 5 - 0.25, 10)

N_q = 1e4  # photons/bit
q = 1.602e-19  # C or A*s or F*V
gamma = 0.53  # A/W
data_rate = 100 * 1e6  # b/s
noise_bandwidth = 2 * data_rate  # Hz
B = noise_bandwidth
# Ibg = 0 # A
Ibg = 5100 * 1e-6  # A
I2 = 0.562
k = 1.38064852 * 1e-23  # J/K
Tk = 295  # K
G = 10
eta = 112 * 1e-8  # F/m^2
A = 1 * 1e-4  # m^2
B_Gamma = 1.5
gm = 30 * 1e-3  # s
I3 = 0.0868

nLed = 60
# nLed = 60
Pt = 0.02  # W
Pt *= nLed * nLed
Hn_value_data = np.array([])

for i in range(ROOM_SIZE[0]):
    for j in range(ROOM_SIZE[1]):
        Hn_value_data = np.append(Hn_value_data,
                                  np.load(r'Hn_value_data/Hn_value_%s.npy' % (str(i) + str(j))))
Hn_value_data = Hn_value_data.reshape(ROOM_SIZE[0], ROOM_SIZE[1], 50, 50)


def plotting(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('noise')
    plt.show()


for i in range(len(xt)):
    for j in range(len(yt)):
        Pr = Pt * Hn_value_data[i][j]
        n_shot = 2 * q * gamma * Pr * B + 2 * q * Ibg * I2 * B
        n_thermal = np.ones((50, 50)) * \
                    ((8 * np.pi * k * Tk * eta * A * I2 * (B ** 2) / G) +
                     ((16 * (np.pi ** 2) * k * Tk * B_Gamma * (eta ** 2) * (A ** 2) * I3 * (B ** 2)) / gm))
        n = N_q * (n_shot + n_thermal)
        # n_shot *= N_q
        # n_thermal *= N_q
        # np.save(r'noise_value_data/noise_value_%s.npy' % (str(i) + str(j)), n.T)
        # np.save(r'n_shot_value_data/n_shot_value_%s.npy' % (str(i) + str(j)), n_shot.T)
        # np.save(r'n_thermal_value_data/n_thermal_value_%s.npy' % (str(i) + str(j)), n_thermal.T)
        if i == 2 and j == 2:
            print('shot noise', np.mean(n_shot))
            print('thermal noise', np.mean(n_thermal))

print('Finish.')
