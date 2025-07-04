import argparse
import subprocess

from plotting import plot_alpha_sweep, plot_eligibility
from plotting.plot_eligibility import get_sparse_lambda_func, get_trunc_lambda_func


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='results')
    args = parser.parse_args()


    ### Figure 2 (right)
    subprocess.run("python run_counterexample.py", shell=True)


    ### Figure 5
    plot_eligibility.main(
        curves=[
            (get_sparse_lambda_func(lambd=0.9, m=1), '#2980b9', "$m=1$"),
            (get_sparse_lambda_func(lambd=0.7518, m=3), '#27ae60', "$m=3$"),
            (get_sparse_lambda_func(lambd=0.6473, m=5), '#c0392b', "$m=5$"),
        ],
        name='eligibility_sparse',
    )


    ### Figure 6
    plot_alpha_sweep.main(
        patterns=[
            f"alpha-{{alpha}}_estimator-lambda-0.9.npy",
            f"alpha-{{alpha}}_estimator-sparse-0.7518-3.npy",
            f"alpha-{{alpha}}_estimator-sparse-0.6473-5.npy",
        ],
        labels=[
            "$m=1$",
            "$m=3$",
            "$m=5$",
        ],
        colors=[
            '#2980b9',
            '#27ae60',
            '#c0392b',
        ],
        name='rw19_sparse',
        input_dir=args.directory,
    )


    ### Figure 7
    plot_eligibility.main(
        curves = [
            (get_trunc_lambda_func(lambd=0.99, L=10), '#3498db', "$\lambda=0.99$, $L=10$"),
            (get_trunc_lambda_func(lambd=0.92, L=20), '#8e44ad', "$\lambda=0.92$, $L=20$"),
            (get_sparse_lambda_func(lambd=0.9, m=1), 'black', "$\lambda=0.9$, $L=\infty$"),
        ],
        name='eligibility_trunc',
    )


    ### Figure 8
    plot_alpha_sweep.main(
        patterns=[
            f"alpha-{{alpha}}_estimator-trunc-0.99-10.npy",
            f"alpha-{{alpha}}_estimator-trunc-0.92-20.npy",
            f"alpha-{{alpha}}_estimator-lambda-0.9.npy",
        ],
        labels = [
            "$\lambda=0.99$, $L=10$",
            "$\lambda=0.92$, $L=20$",
            "$\lambda=0.9$, $L=\infty$",
        ],
        colors = [
            '#3498db',
            '#8e44ad',
            'black',
        ],
        name='rw19_trunc',
        input_dir=args.directory,
    )
