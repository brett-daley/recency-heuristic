import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_power

from plotting import save, set_plot_size


def quiver(operator, x, y):
    X, Y = np.meshgrid(x, y)
    dx = np.zeros_like(X)
    dy = np.zeros_like(Y)

    for i in range(len(x)):
        for j in range(len(y)):
            p = np.array([X[i,j], Y[i,j]])
            g = (operator(p) - p) / (np.linalg.norm(operator(p) - p) + 1e-6)
            dx[i,j] = g[0]
            dy[i,j] = g[1]

    plt.gca().quiver(X, Y, dx, dy, zorder=1)


if __name__ == '__main__':
    matplotlib.style.use("styles/counterexample.mplstyle")
    plt.figure()
    plt.plot([-2, 2], [-2, 2], linestyle='--', linewidth=0.25, color='tab:blue', zorder=0)  # 45-degree angle
    plt.plot(0, 0, marker='*', color='red')  # Fixed point

    r = np.array([0, 0])
    p = 0.4
    P_pi = np.array([[p, 1-p], [1-p, p]])
    discount = 0.9
    v_pi = r / (1 - discount)

    Bellman = lambda v: r + (discount * P_pi) @ v

    tau = 1  # Delay
    operator = lambda v: v + matrix_power(discount * P_pi, tau) @ (Bellman(v) - v)

    x = y = np.linspace(-1.5, 1.5, 12)
    quiver(operator, x, y)

    plt.xlabel("$V_0(s_1)$")
    plt.ylabel("$V_0(s_2)$")
    plt.xlim([-1.6, 1.6])
    plt.ylim([-1.6, 1.6])

    set_plot_size(aspect=1)
    save("divergence", directory="plots", pdf=False)
    save("divergence", directory="plots", pdf=True)
