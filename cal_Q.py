"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc: verify the point of Q below 1e-6
"""
from scipy.special import erf, erfc
import numpy as np


for Q in np.linspace(4, 5, 1000):
    if 0.5 * (1 - erf((Q / (2 ** 0.5)))) < 1e-6:
        print(Q)
        break
