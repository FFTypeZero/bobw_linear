import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import run_trials_in_parallel, compute_gap


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


def run_change_T(algos, n_trials=1000):
    d = 10
    omega = 0.1
    noise_level = 1.0
    Ts = np.array([10 + 1000 * i for i in range(7)])

    results_total = np.zeros((len(algos), len(Ts), n_trials))
    min_gaps = np.zeros(len(Ts))
    np.random.seed(6)

    for i, T in enumerate(Ts):
        X, thetas = get_soare_instance(d, T, omega)
        gap, opt_arm = compute_gap(X, thetas)
        min_gaps[i] = gap

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, 6)
            results_total[j][i] = np.array(results)
            np.savez_compressed(f'plot_data/{algo}/{algo}_results_soare_sto.npz', 
                                results=results_total[j], Ts=Ts, min_gaps=min_gaps)

    return results_total, min_gaps


def get_plot(algos):
    fig, axs = plt.subplots(1, 1)
    for algo in algos:
        loaded = np.load(f'plot_data/{algo}/{algo}_results_soare_sto.npz')
        results = loaded['results']
        Ts = loaded['Ts']
        error_prob = 1.0 - np.mean(results, axis=1)
        confi_bound = 1.96 * np.std(results, axis=1) / np.sqrt(results.shape[1])
        axs.plot(Ts, error_prob, 'o-', label=algo)
        axs.fill_between(Ts, error_prob - confi_bound, error_prob + confi_bound, alpha=0.4)
    axs.set_xlabel('$T$')
    axs.set_ylabel('error probability')
    axs.set_yscale('log')
    axs.set_title('Error Probability vs. $T$ Under Soare et al. (2014) Examples')
    axs.legend(loc='best')
    axs.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run stationary experiments")
    parser.add_argument("-r", "--run", type=int, default=1,
                        help="Whether to run the experiments or just plot the results")
    args = parser.parse_args()
    run = args.run

    n_trials = 1000
    algos = ['G-BAI', 'Peace', 'P1-Peace', 'P1-RAGE']

    if run:
        results, min_gaps = run_change_T(algos, n_trials)
        for j, algo in enumerate(algos):
            print(f"{algo} Oscillation magnitude accuracy: {np.mean(results[j], axis=1)}")
        print(f"Oscillation magnitude minimum gaps: {min_gaps}")

    get_plot(algos)