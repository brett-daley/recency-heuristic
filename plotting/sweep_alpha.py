from argparse import ArgumentParser
from functools import partial
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import calculate_auc, save, set_plot_size


def alpha_sweep(input_dir, output_dir, alpha_values, patterns, labels, colors, use_ylabel=True, name=None):
    plt.figure()

    for p, label, color in zip(patterns, labels, colors):
        p = os.path.join(input_dir, p)

        means, conf95s, _ = zip(*[
            calculate_auc(p.format(alpha=a), reduce='mean')
            for a in alpha_values
        ])

        xs = alpha_values
        ys, conf95s = map(np.array, [means, conf95s])

        # Plot mean
        plt.plot(xs, ys, label=label, color=color)
        # Plot dashed horizontal line for minimum error
        min_value = np.full_like(ys, fill_value=np.min(ys))
        plt.plot(xs, min_value, linestyle='--', linewidth=1, color=color)
        # Shade 95% confidence interval
        plt.fill_between(xs, (ys - conf95s), (ys + conf95s), alpha=0.25, linewidth=0, color=color)

    plt.xlabel(r"$\alpha$")
    if use_ylabel:
        plt.ylabel("Average RMS error")
    else:
        plt.ylabel(" ")

    plt.xlim([0, 1])
    plt.ylim([0.25, 0.55])

    set_plot_size(aspect=1)
    plt.legend(loc="best")
    save(name, output_dir, pdf=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()

    matplotlib.style.use('custom.mplstyle')

    alpha_values = [
        0.0067, 0.0074, 0.0082, 0.0091, 0.0101, 0.0111, 0.0123, 0.0136, 0.015, 0.0166,
        0.0183, 0.0202, 0.0224, 0.0247, 0.0273, 0.0302, 0.0334, 0.0369, 0.0408, 0.045,
        0.0498, 0.055, 0.0608, 0.0672, 0.0743, 0.0821, 0.0907, 0.1003, 0.1108, 0.1225,
        0.1353, 0.1496, 0.1653, 0.1827, 0.2019, 0.2231, 0.2466, 0.2725, 0.3012, 0.3329,
        0.3679, 0.4066, 0.4493, 0.4966, 0.5488, 0.6065, 0.6703, 0.7408, 0.8187, 0.9048, 1]

    plot_func = partial(alpha_sweep, args.input_dir, args.output_dir, alpha_values)

    plot_func(
        patterns=[
            # f"alpha-{{alpha}}_estimator-nstep_1.npy",
            # f"alpha-{{alpha}}_estimator-nstep_2.npy",
            # f"alpha-{{alpha}}_estimator-nstep_3.npy",
            # f"alpha-{{alpha}}_estimator-nstep_4.npy",
            # f"alpha-{{alpha}}_estimator-nstep_5.npy",
            # f"alpha-{{alpha}}_estimator-nstep_6.npy",
            # f"alpha-{{alpha}}_estimator-nstep_7.npy",
            # f"alpha-{{alpha}}_estimator-nstep_8.npy",
            # f"alpha-{{alpha}}_estimator-nstep_10.npy",
            # f"alpha-{{alpha}}_estimator-lambda_0.5.npy",
            # f"alpha-{{alpha}}_estimator-lambda_0.6.npy",
            # f"alpha-{{alpha}}_estimator-lambda_0.7.npy",
            # f"alpha-{{alpha}}_estimator-lambda_0.8.npy",
            # f"alpha-{{alpha}}_estimator-lambda_0.9.npy",
            # 
            f"alpha-{{alpha}}_estimator-trunc_0.99_10.npy",
            f"alpha-{{alpha}}_estimator-trunc_0.92_20.npy",
            f"alpha-{{alpha}}_estimator-lambda_0.9.npy",
            # 
            # f"alpha-{{alpha}}_estimator-lambda_0.9.npy",
            # f"alpha-{{alpha}}_estimator-space_3_0.7518.npy",
            # f"alpha-{{alpha}}_estimator-space_5_0.6473.npy",
        ],
        labels=[
            "$\lambda=0.99$, $L=10$",
            "$\lambda=0.92$, $L=20$",
            "$\lambda=0.9$, $L=\infty$",
            # 
            # "$m=1$",
            # "$m=3$",
            # "$m=5$",
        ],
        # colors=['#2980b9', '#27ae60', '#c0392b'],
        colors=['#3498db', '#8e44ad', 'black'],
        use_ylabel=(0 == 0),
        name=f"rw19_super"
    )
