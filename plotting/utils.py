import os

import matplotlib.pyplot as plt
import numpy as np


def calculate_auc(path, reduce='sum', exclude_first_n=0):
    assert reduce in {'sum', 'mean'}
    performance = np.load(path)
    assert performance.ndim == 2

    performance = performance[:, exclude_first_n:]
    if reduce == 'sum':
        AUCs = np.sum(performance, axis=1)
    else:
        AUCs = np.mean(performance, axis=1)
    mean_auc = np.mean(AUCs)

    conf95_auc = 0.0
    n = len(AUCs)
    if n > 1:
        std_auc = np.std(AUCs, ddof=1)
        conf95_auc = 1.96 * std_auc / np.sqrt(n)

    return mean_auc, conf95_auc, n


def save(name, directory, pdf):
    if not os.path.exists(directory):
        os.mkdir(directory)

    path = os.path.join(directory, name)
    if pdf:
        path += '.pdf'
        plt.savefig(path, format='pdf')
    else:
        path += '.png'
        plt.savefig(path, format='png')
    print(f"Saved plot as {path}", flush=True)


def set_plot_size(aspect=1):
    ax = plt.gca()
    ax.set_aspect(1.0 / (aspect * ax.get_data_ratio()))

    fig = plt.gcf()
    y = 4.8
    x = aspect * y
    fig.set_size_inches(x, y)

    if aspect > 1:
        fig.tight_layout(pad=0)
    else:
        fig.tight_layout(pad=0.1)
