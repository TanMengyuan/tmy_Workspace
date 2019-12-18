"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@time: 2019/12/17 18:37
@desc: plot the activation function
"""

import numpy as np
import matplotlib.pyplot as plt


class Sigmoid:
    name = "Sigmoid function"

    def f(self, x):
        return 1 / (1 + np.exp(-x))


class Tanh:
    name = "Tanh function"

    def f(self, x):
        return np.tanh(x)


class Relu:
    name = "ReLU function"

    def f(self, x):
        return np.maximum(np.zeros(x.size), x)


class Softmax:
    """
    Softmax function can't plot
    """
    name = "Softmax function"

    def f(self, x):
        return 0


if __name__ == '__main__':
    x = np.linspace(-10, 10, 1000)
    func_list = [Sigmoid(), Tanh(), Relu()]  # no softmax

    for func in func_list:
        plt.cla()
        y = func.f(x)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(func.name)
        plt.plot(x, y)
        plt.show()
        # plt.savefig("./activation_function_fig/%s.jpg" % func.name)
