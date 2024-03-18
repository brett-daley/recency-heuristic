import os

import matplotlib.pyplot as plt


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
