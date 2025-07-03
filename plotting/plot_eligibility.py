import math

import matplotlib.pyplot as plt
import numpy as np

from plotting.utils import save, set_plot_size


def get_sparse_lambda_func(lambd, m):
    def sparse_lambda(i):
        assert i >= 0
        return pow(lambd, math.floor((i + m - 1) / m))
    return sparse_lambda


def get_trunc_lambda_func(lambd, L):
    assert L >= 1
    def trunc_lambda(i):
        assert i >= 0
        if i < L:
            return pow(lambd, i)
        else:
            return 0.0
    return trunc_lambda


def main(curves, name, discount=1.0):
    plt.style.use('styles/lineplot.mplstyle')
    plt.figure()

    N = 40
    x_axis = np.arange(N + 1)
    plt.xlim(0, N)

    # (func, color, label)
    for weight_func, color, label in curves:
        weights = np.array([pow(discount, i) * weight_func(i) for i in range(N + 1)])
        plt.plot(x_axis, weights, linestyle='--', linewidth=0.5, marker='.', color=color, label=label)

    margin = 0.01
    plt.ylim(-margin, 1 + margin)

    plt.xlabel("Time since state visitation")
    plt.ylabel("Eligibility")

    plt.legend(loc="upper right")

    set_plot_size()
    directory = 'figures'
    for pdf in [False, True]:
        save(name, directory, pdf)
