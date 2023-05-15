import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import run_trials_in_parallel, compute_gap
from run_adv_multi import get_sto_multi


def get_malicious_instance(d, T, omega):
    """
    Malicious example to fail Peace
    """
    X = np.eye(d)
    x_extra = np.zeros(d)
    x_extra[0] = np.cos(omega)
    x_extra[1] = np.sin(omega)
    X = np.vstack((X, x_extra))

    theta1 = np.ones(d)
    theta1[0] = 0
    theta2 = np.zeros(d)
    theta2[0] = 2.0

    thetas = np.zeros((T, d))
    thetas[:int(T/3), :] = theta1
    thetas[int(T/3)+1:, :] = theta2

    return X, thetas


def run_malicious(algos, n_trials=1000):
    d = 10
    omega = 0.5
    T = 10000
    noise_level = 1.0

    results_total = np.zeros((len(algos), n_trials))
    np.random.seed(6)

    X, thetas = get_malicious_instance(d, T, omega)

    gap, opt_arm = compute_gap(X, thetas)

    for j, algo in enumerate(algos):
        results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, 6)
        results_total[j] = np.array(results)
        np.savez_compressed(f'plot_data/Single/single_results_malicious.npz', 
                            results=results_total, algos=algos)

    return results_total


def run_sto_multi(algos, n_trials=1000):
    T = 10000
    noise_level = 0.3
    D = 4
    results_total = np.zeros((len(algos), n_trials))

    # Make sure the instances are the same for each run
    np.random.seed(6)
    X, thetas = get_sto_multi(D, T)

    gap, opt_arm = compute_gap(X, thetas)

    for j, algo in enumerate(algos):
        results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, 6)
        results_total[j] = np.array(results)
        np.savez_compressed(f'plot_data/Single/single_results_multi.npz', 
                            results=results_total[j], algos=algos)

    return results_total


def get_plot(case, title):
    loaded = np.load(f'plot_data/Single/single_results_{case}.npz')
    results = loaded['results']
    algos = loaded['algos']
    error_prob = 1.0 - np.mean(results, axis=1)
    confi_bound = 1.96 * np.std(results, axis=1) / np.sqrt(results.shape[1])
    for i in range(len(algos)):
        rects1 = plt.bar(i, error_prob[i], width=0.5, alpha=0.8, yerr=confi_bound[i], capsize=7)

    plt.ylabel('error probability')
    plt.xticks(range(len(algos)), algos)
    plt.title(title)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run non-stationary experiments")
    parser.add_argument("-r", "--run", type=int, default=1,
                        help="Whether to run the experiments or just plot the results")
    args = parser.parse_args()
    run = args.run

    n_trials = 1000
    algos = ['G-BAI', 'Peace', 'P1-Peace', 'P1-RAGE', 'OD-LinBAI', 'Mixed-Peace']

    if run:
        # results_malicious = run_malicious(algos, n_trials)
        results_multi = run_sto_multi(algos, n_trials)
        # for j, algo in enumerate(algos):
            # print(f"{algo} malicious accuracy: {np.mean(results_malicious[j])}")
        for j, algo in enumerate(algos):
            print(f"{algo} multi accuracy: {np.mean(results_multi[j])}")
    
    # get_plot('malicious', 'Experiments under Malicious Example')
    get_plot('multi', 'Experiments under Multivariate Testing Example')
