import os
import time
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from utils import run_trials_in_parallel, compute_gap

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


def get_adv_Soare(d, T, omega, osci_mag, move_gap):
    """
    Soare et al. (2014) example with oscillating arm
    """

    X = np.eye(d)
    x_extra = np.zeros(d)
    x_extra[0] = np.cos(omega)
    x_extra[1] = np.sin(omega)
    X = np.vstack((X, x_extra))

    ts = np.arange(T)
    thetas = np.zeros((T, d))
    thetas[:, 0] = 0.3
    thetas[:, -1] = - osci_mag * np.sin(2 * ts * np.pi / move_gap) + 0.5

    gap, opt_arm = compute_gap(X, thetas)
    assert opt_arm == d - 1

    return X, thetas


def run_change_osci(algos, n_trials=1000, save=True):
    d = 10
    omega = 0.5
    T = 10000
    noise_level = 1.0
    move_gap = 200
    osci_mags = [1.0 * i for i in range(10)]
    min_gaps = np.zeros(len(osci_mags))

    results_total = np.zeros((len(algos), len(osci_mags), n_trials))
    np.random.seed(6)

    for i, osci_mag in enumerate(osci_mags):
        X, thetas = get_adv_Soare(d, T, omega, osci_mag, move_gap)

        gap, opt_arm = compute_gap(X, thetas)
        min_gaps[i] = gap

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, n_workers=6, setting_para=osci_mag)
            results_total[j][i] = np.array(results)

            if save:
                if not os.path.exists(f'plot_data/{algo}'):
                    os.makedirs(f'plot_data/{algo}')
                np.savez_compressed(f'plot_data/{algo}/{algo}_results_soare_osci.npz', 
                                    results=results_total[j], osci_mags=osci_mags, min_gaps=min_gaps)

    return results_total, min_gaps


def run_change_period(algos, n_trials=1000, save=True):
    d = 10
    omega = 0.5
    T = 10000
    noise_level = 1.0
    osci_mag = 2.5
    move_gaps = [300 + 300 * i for i in range(10)]
    min_gaps = np.zeros(len(move_gaps))

    results_total = np.zeros((len(algos), len(move_gaps), n_trials))
    np.random.seed(6)

    for i, move_gap in enumerate(move_gaps):
        X, thetas = get_adv_Soare(d, T, omega, osci_mag, move_gap)

        gap, opt_arm = compute_gap(X, thetas)
        min_gaps[i] = gap

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, n_workers=6, setting_para=move_gap)
            results_total[j][i] = np.array(results)

            if save:
                if not os.path.exists(f'plot_data/{algo}'):
                    os.makedirs(f'plot_data/{algo}')
                np.savez_compressed(f'plot_data/{algo}/{algo}_results_soare_period.npz', 
                                    results=results_total[j], move_gaps=move_gaps, min_gaps=min_gaps)

    return results_total, min_gaps


def get_plot(algos):
    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [2, 1]}, figsize=(11, 7.5))
    for j, algo in enumerate(algos):
        loaded_osci = np.load(f'plot_data/{algo}/{algo}_results_soare_osci.npz')
        loaded_period = np.load(f'plot_data/{algo}/{algo}_results_soare_period.npz')
        results_osci = loaded_osci['results']
        results_period = loaded_period['results']
        osci_mags = loaded_osci['osci_mags']
        move_gaps = loaded_period['move_gaps']
        min_gaps_osci = loaded_osci['min_gaps']
        min_gaps_period = loaded_period['min_gaps']

        error_prob_osci = 1.0 - np.mean(results_osci, axis=1)
        error_prob_period = 1.0 - np.mean(results_period, axis=1)
        confi_bound_osci = 1.96 * np.std(results_osci, axis=1) / np.sqrt(results_osci.shape[1])
        confi_bound_period = 1.96 * np.std(results_period, axis=1) / np.sqrt(results_period.shape[1])

        axs[0, 0].plot(osci_mags, error_prob_osci, 'o-', label=algo)
        axs[0, 0].fill_between(osci_mags, error_prob_osci - confi_bound_osci, error_prob_osci + confi_bound_osci, alpha=0.4)
        axs[0, 1].plot(move_gaps, error_prob_period, 'o-', label=algo)
        axs[0, 1].fill_between(move_gaps, error_prob_period - confi_bound_period, error_prob_period + confi_bound_period, alpha=0.4)
        if j == 0:
            axs[1, 0].plot(osci_mags, min_gaps_osci, 'o-', color='black', label='minimum gap')
            axs[1, 1].plot(move_gaps, min_gaps_period, 'o-', color='black')
    axs[1, 0].set_xlabel('oscillation magnitude ($s$)')
    axs[1, 1].set_xlabel('oscillation period ($L$)')
    axs[0, 0].set_ylabel('error probability')
    axs[1, 0].set_ylabel('minimum gap')
    axs[0, 0].set_title('Error Probability vs. Oscillation Magnitude')
    axs[0, 1].set_title('Error Probability vs. Oscillation Period')
    axs[1, 0].set_ylim([0, 0.4])
    axs[1, 0].legend(loc='best')
    for i in range(2):
        # axs[0, i].set_ylim([-0.05, 1.05])
        axs[0, i].set_yscale('log')
        axs[0, i].grid(True)
    axs[0, 0].legend(loc='center', bbox_to_anchor=(0.5,0), bbox_transform=fig.transFigure, ncol=len(algos))
    axs[1, 0].grid(True)
    axs[1, 1].grid(True)

    # plt.suptitle("Experiments under Non-stationary Soare et al. (2014)'s Example")
    plt.savefig('figs/adv_soare_log.pdf', bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run non-stationary experiments")
    parser.add_argument("-r", "--run", type=int, default=0,
                        help="Whether to run the experiments or just plot the results")
    args = parser.parse_args()
    run = args.run

    save = True
    n_trials = 1000
    algos = ['G-BAI', 'Peace', 'P1-Peace', 'P1-RAGE', 'OD-LinBAI', 'Mixed-Peace']

    start_time = time.time()
    if run:
        results_osci, min_gaps_osci = run_change_osci(algos, n_trials, save)
        results_period, min_gaps_period = run_change_period(algos, n_trials, save)
        for j, algo in enumerate(algos):
            print(f"{algo} Oscillation magnitude accuracy: {np.mean(results_osci[j], axis=1)}")
        print(f"Oscillation magnitude minimum gaps: {min_gaps_osci}")
        for j, algo in enumerate(algos):
            print(f"{algo} Oscillation period accuracy: {np.mean(results_period[j], axis=1)}")
        print(f"Oscillation period minimum gaps: {min_gaps_period}")

    end_time = time.time()
    print(f"Total running time: {end_time - start_time}")
    get_plot(algos)
