"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@time: 2020/5/8 22:46
"""

import numpy as np

a = np.load("time_saver_DL.npy")
a = a[4000:4000 + 60]
np.save("time_saver_DL_part.npy", a)
print(len(a))
