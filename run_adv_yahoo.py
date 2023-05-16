import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import run_trials_in_parallel, compute_gap


def add_repeatition(thetas, T, L):
    try:
        T % (7 * L) == 0
    except:
        raise ValueError("T must be divisible by L")

    r = int(T / (7 * L))
    thetas_period = np.repeat(thetas, L, axis=0)
    thetas_final = np.vstack([thetas_period] * r)
    return thetas_final


def run_change_duration(algos, n_trials=1000, save=True):
    T = 21000
    noise_level = 1.0

    durations = [100, 200, 500, 1000, 1500, 3000]
    min_gaps = np.zeros(len(durations))
    results_total = np.zeros((len(algos), len(durations), n_trials))

    np.random.seed(6)
    loaded = np.load('yahoo_data_pca.npz')
    X = loaded['Xs']
    thetas_single = loaded['thetas']

    for i, duration in enumerate(durations):
        thetas = add_repeatition(thetas_single, T, duration)

        gap, opt_arm = compute_gap(X, thetas)
        min_gaps[i] = gap

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, n_workers=6, setting_para=duration)
            results_total[j][i] = np.array(results)

            if save:
                if not os.path.exists(f'plot_data/{algo}'):
                    os.makedirs(f'plot_data/{algo}')
                np.savez_compressed(f'plot_data/{algo}/{algo}_results_yahoo_pca_repeat.npz', 
                                    results=results_total[j], durations=durations, min_gaps=min_gaps)

    return results_total, min_gaps

def get_plot(algos):
    fig, axs = plt.subplots(1, 1)
    for algo in algos:
        loaded = np.load(f'plot_data/{algo}/{algo}_results_yahoo_pca_repeat.npz')
        results = loaded['results']
        durations = loaded['durations']

        error_prob = 1.0 - np.mean(results, axis=1)
        confi_bound = 1.96 * np.std(results, axis=1) / np.sqrt(results.shape[1])

        axs.plot(durations, error_prob, 'o-', label=algo)
        axs.fill_between(durations, error_prob - confi_bound, error_prob + confi_bound, alpha=0.4)
    axs.set_xlabel('repeat durations')
    axs.set_ylabel('error probability')
    axs.set_ylim([-0.05, 1.05])
    axs.legend(loc='best')
    axs.grid(True)
    axs.set_title('Experiments under Yahoo! News Article Example')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run non-stationary experiments")
    parser.add_argument("-r", "--run", type=int, default=1,
                        help="Whether to run the experiments or just plot the results")
    args = parser.parse_args()
    run = args.run

    save = True
    n_trials = 1000
    algos = ['G-BAI', 'Peace', 'P1-Peace', 'P1-RAGE', 'OD-LinBAI', 'Mixed-Peace']
    # algos = ['G-BAI', 'Peace', 'P1-Peace', 'OD-LinBAI']

    if run:
        results_duration, min_gaps_duration = run_change_duration(algos, n_trials, save)
        for j, algo in enumerate(algos):
            print(f"{algo} Repeat duration accuracy: {np.mean(results_duration[j], axis=1)}")
        print(f"Repeat duration minimum gaps: {min_gaps_duration}")

    get_plot(algos)
