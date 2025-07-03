import math

import matplotlib.pyplot as plt
import numpy as np

from utils import save, set_plot_size


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


def main(discount=1.0):
    N = 40
    x_axis = np.arange(N + 1)
    plt.xlim(0, N)

    # (func, color, label)
    curves = [
        (get_sparse_lambda_func(lambd=0.9, m=1), '#2980b9', "$m=1$"),
        (get_sparse_lambda_func(lambd=0.7518, m=3), '#27ae60', "$m=3$"),
        (get_sparse_lambda_func(lambd=0.6473, m=5), '#c0392b', "$m=5$"),
    ]
    # curves = [
    #     (get_trunc_lambda_func(lambd=0.99, L=10), '#3498db', "$\lambda=0.99$, $L=10$"),
    #     (get_trunc_lambda_func(lambd=0.92, L=20), '#8e44ad', "$\lambda=0.92$, $L=20$"),
    #     (get_sparse_lambda_func(lambd=0.9, m=1), 'black', "$\lambda=0.9$, $L=\infty$"),
    # ]

    for weight_func, color, label in curves:
        weights = np.array([pow(discount, i) * weight_func(i) for i in range(N + 1)])
        plt.plot(x_axis, weights, linestyle='--', linewidth=0.5, marker='.', color=color, label=label)

    margin = 0.01
    plt.ylim(-margin, 1 + margin)

    plt.xlabel("Time since state visitation")
    plt.ylabel("Eligibility")

    plt.legend(loc="upper right")

    set_plot_size()
    name = 'eligibility'
    directory = 'figures'
    save(name, directory, pdf=False)
    save(name, directory, pdf=True)


if __name__ == '__main__':
    plt.style.use('styles/lineplot.mplstyle')
    main()
