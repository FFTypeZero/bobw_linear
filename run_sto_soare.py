import os
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from utils import run_trials_in_parallel_soare, compute_gap

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


def get_soare_instance(d, T, omega):
    """"
    Standard Soare et al. (2014) example
    """
    X = np.eye(d)
    x_extra = np.zeros(d)
    x_extra[0] = np.cos(omega)
    x_extra[1] = np.sin(omega)
    X = np.vstack((X, x_extra))

    theta = np.zeros(d)
    theta[0] = 2.0

    thetas = np.zeros((T, d))
    thetas[:] = theta

    return X, thetas


def run_change_T(algos, n_trials=1000, save=True):
    d = 10
    omega = 0.1
    noise_level = 1.0
    Ts = np.array([10 + 1000 * i for i in range(6)])

    results_total = np.zeros((len(algos), len(Ts), n_trials))
    min_gaps = np.zeros(len(Ts))

    for i, T in enumerate(Ts):
        X, thetas = get_soare_instance(d, T, omega)
        gap, opt_arm = compute_gap(X, thetas)
        min_gaps[i] = gap

        for j, algo in enumerate(algos):
            saving_dir = f'plot_data/{algo}/soare_sto'
            saving_file = f'{saving_dir}/results_T={T}.npz'
            results = run_trials_in_parallel_soare(n_trials, X, T, thetas, opt_arm, algo, saving_dir, saving_file, save=save, noise_level=noise_level, n_workers=6, setting_para=T)
            results_total[j][i] = np.array(results)

    return results_total, min_gaps


def get_plot(algos):
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    for algo in algos:
        Ts = []
        error_prob = []
        confi_bound = []
        for file in os.listdir(f'plot_data/{algo}/soare_sto'):
            if file.endswith('.npz'):
                loaded = np.load(f'plot_data/{algo}/soare_sto/{file}')
                results = loaded['results']
                error_prob.append(1.0 - np.mean(results))
                confi_bound.append(1.96 * np.std(results) / np.sqrt(results.shape[0]))
                Ts.append(loaded['T'])

        Ts = np.array(Ts)
        error_prob = np.array(error_prob)
        confi_bound = np.array(confi_bound)
        sort_idx = np.argsort(Ts)
        Ts = Ts[sort_idx]
        error_prob = error_prob[sort_idx]
        confi_bound = confi_bound[sort_idx]
        axs.plot(Ts, error_prob, 'o-', label=algo)
        axs.fill_between(Ts, error_prob - confi_bound, error_prob + confi_bound, alpha=0.4)
    axs.set_xlabel('budget ($T$)')
    axs.set_ylabel('error probability')
    axs.set_yscale('log')
    axs.set_title("Error Probability vs. $T$ Under Soare et al. (2014)'s Example")
    axs.legend(loc='best')
    axs.grid(True)
    plt.savefig('figs/sto_soare_log.pdf', bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run stationary experiments")
    parser.add_argument("-r", "--run", type=int, default=0,
                        help="Whether to run the experiments or just plot the results")
    args = parser.parse_args()
    run = args.run

    save = True
    n_trials = 20000
    algos = ['G-BAI', 'Peace', 'P1-Peace', 'P1-RAGE', 'OD-LinBAI', 'Mixed-Peace']

    if run:
        results, min_gaps = run_change_T(algos, n_trials, save)
        for j, algo in enumerate(algos):
            print(f"{algo} accuracy: {np.mean(results[j], axis=1)}")
        print(f"Minimum gaps: {min_gaps}")

    get_plot(algos)
