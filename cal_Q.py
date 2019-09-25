from scipy.special import erf, erfc
import numpy as np


for Q in np.linspace(4, 5, 1000):
    if 0.5 * (1 - erf((Q / (2 ** 0.5)))) < 1e-3:
        print(Q)
        break
