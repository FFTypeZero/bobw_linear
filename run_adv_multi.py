import os
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from utils import run_trials_in_parallel, compute_gap


def get_sto_multi(D, T, num_sparse=None):
    """
    Multivariate testing example from Fiez et al. (2019)
    """
    alpha1 = 1
    alpha2 = 0.5
    variants = list(range(D))
    individual_index_dict = {}

    count = 0
    for key in variants:
        individual_index_dict[key] = count
        count += 1

    pairwise_index_dict = {}
    count = 0
    pairs = []
    for pair in itertools.combinations(range(D), 2):
        pairs.append(pair)
        key1 = pair[0]
        key2 = pair[1]
        pairwise_index_dict[(key1, key2)] = count
        count += 1

    individual_offset = 1
    pairwise_offset = 1 + len(individual_index_dict)
    num_features = 1 + len(individual_index_dict) + len(pairwise_index_dict)
    num_arms = 2**D

    combinations = list(itertools.product([-1, 1], repeat=D))

    X = -np.ones((num_arms, num_features))

    for idx in range(num_arms):
        bias_feature_index = [0]
        individual_feature_index = [individual_offset + individual_index_dict[i] for i, val in enumerate(combinations[idx]) if val == 1]
        pairwise_feature_index = [pairwise_offset + pairwise_index_dict[pair] for pair in pairs if combinations[idx][pair[0]] == combinations[idx][pair[1]]]
        feature_index = bias_feature_index + individual_feature_index + pairwise_feature_index
        X[idx, feature_index] = 1

    while True:
        theta_star = np.random.randint(-10, 10, (num_features, 1))/100
        theta_star[individual_offset:pairwise_offset] = alpha1*theta_star[individual_offset:pairwise_offset]
        theta_star[pairwise_offset] = alpha2*theta_star[pairwise_offset]

        if num_sparse:
            sparse_index = np.zeros(D)
            sparse_index[np.random.choice(len(sparse_index), num_sparse, replace=False)] = 1
            bias_feature_index = [0]
            individual_feature_index = [individual_offset + individual_index_dict[i] for i, val in enumerate(sparse_index) if val == 1]
            pairwise_feature_index = [pairwise_offset + pairwise_index_dict[pair] for pair in pairs if sparse_index[pair[0]] == 1 and sparse_index[pair[1]] == 1]
            feature_index = bias_feature_index + individual_feature_index + pairwise_feature_index
            theta_star[~np.array(feature_index)] = 0

        rewards = (X@theta_star).reshape(-1)
        top_rewards = sorted(rewards, reverse=True)[:2]
        
        if top_rewards[0] - top_rewards[1] < 10e-6:
            continue
        else:
            break

    thetas = np.zeros((T, num_features))
    thetas[:] = theta_star.reshape(-1)

    return X, thetas


def add_perturbation_multi(X, theta_stars, osci_mag, move_gap):
    T, num_features = theta_stars.shape
    theta_star = theta_stars[0]

    ts = np.arange(T)
    max_mag = np.max(np.abs(theta_star))

    gap, opt_arm = compute_gap(X, theta_star)
    thetas = np.zeros((T, num_features))
    count = 0
    while True:
        for j in range(num_features):
            oscillate = np.random.randint(0, 2)
            if oscillate // 2 == 0:
                phase_shift = np.random.uniform(0, 2 * np.pi)
                thetas[:, j] = osci_mag * max_mag * np.sin(2 * ts * np.pi / move_gap + phase_shift) + theta_star[j]

        gap_adv, opt_arm_adv = compute_gap(X, thetas)

        if opt_arm_adv == opt_arm:
            break
        else:
            count += 1
            if count > 50:
                print(f"Current osci_mag: {osci_mag}, move_gap: {move_gap}")
                print('Warning: too many trials to generate non-stationary example')
                assert False

    return thetas


def run_change_osci(algos, n_trials=1000, save=True):
    T = 10000
    noise_level = 0.3
    osci_mags = [1.0 * i for i in range(10)]
    min_gaps = np.zeros(len(osci_mags))

    move_gap = 900
    D = 4
    results_total = np.zeros((len(algos), len(osci_mags), n_trials))

    # Make sure the instances are the same for each run
    np.random.seed(6)
    thetas_all = []
    X, theta_stars = get_sto_multi(D, T)
    for osci_mag in osci_mags:
        thetas_temp = add_perturbation_multi(X, theta_stars, osci_mag, move_gap)
        thetas_all.append(thetas_temp)

    for i, osci_mag in enumerate(osci_mags):
        thetas = thetas_all[i]

        gap, opt_arm = compute_gap(X, thetas)
        min_gaps[i] = gap

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, n_workers=6, setting_para=osci_mag)
            results_total[j][i] = np.array(results)

            if save:
                if not os.path.exists(f'plot_data/{algo}'):
                    os.makedirs(f'plot_data/{algo}')
                np.savez_compressed(f'plot_data/{algo}/{algo}_results_multi_osci.npz', 
                                    results=results_total[j], osci_mags=osci_mags, min_gaps=min_gaps)

    return results_total, min_gaps


def run_change_period(algos, n_trials=1000, save=True):
    T = 10000
    noise_level = 0.3
    osci_mag = 2.0

    move_gaps = [300 + 300 * i for i in range(10)]
    min_gaps = np.zeros(len(move_gaps))
    D = 4
    results_total = np.zeros((len(algos), len(move_gaps), n_trials))

    # Make sure the instances are the same for each run
    np.random.seed(6)
    theta_all = []
    X, theta_stars = get_sto_multi(D, T)
    for move_gap in move_gaps:
        theta_temp = add_perturbation_multi(X, theta_stars, osci_mag, move_gap)
        theta_all.append(theta_temp)

    for i, move_gap in enumerate(move_gaps):
        thetas = theta_all[i]

        gap, opt_arm = compute_gap(X, thetas)
        min_gaps[i] = gap

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, n_workers=6, setting_para=move_gap)
            results_total[j][i] = np.array(results)

            if save:
                if not os.path.exists(f'plot_data/{algo}'):
                    os.makedirs(f'plot_data/{algo}')
                np.savez_compressed(f'plot_data/{algo}/{algo}_results_multi_period.npz', 
                                    results=results_total[j], move_gaps=move_gaps, min_gaps=min_gaps)

    return results_total, min_gaps


def get_plot(algos):
    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [2, 1]}, figsize=(11, 7.5))
    for j, algo in enumerate(algos):
        loaded_osci = np.load(f'plot_data/{algo}/{algo}_results_multi_osci.npz')
        loaded_period = np.load(f'plot_data/{algo}/{algo}_results_multi_period.npz')
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
    axs[1, 0].set_xlabel('oscillation scale ($s$)')
    axs[1, 1].set_xlabel('oscillation period ($L$)')
    axs[0, 0].set_ylabel('error probability')
    axs[1, 0].set_ylabel('minimum gap')
    axs[0, 0].set_title('Error Probability vs. Oscillation Scale')
    axs[0, 1].set_title('Error Probability vs. Oscillation Period')
    axs[1, 0].legend(loc='best')
    for i in range(2):
        # axs[0, i].set_ylim([-0.05, 1.05])
        axs[0, i].set_yscale('log')
        axs[0, i].grid(True)
    axs[0, 0].legend(loc='lower left', bbox_to_anchor=(0.15,0), bbox_transform=fig.transFigure, ncol=len(algos))
    axs[1, 0].grid(True)
    axs[1, 1].grid(True)

    # plt.suptitle("Experiments under Non-stationary Multivariate Testing Example")
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

    if run:
        results_osci, min_gaps_osci = run_change_osci(algos, n_trials, save)
        results_period, min_gaps_period = run_change_period(algos, n_trials, save)
        for j, algo in enumerate(algos):
            print(f"{algo} Oscillation magnitude accuracy: {np.mean(results_osci[j], axis=1)}")
        print(f"Oscillation magnitude minimum gaps: {min_gaps_osci}")
        for j, algo in enumerate(algos):
            print(f"{algo} Oscillation period accuracy: {np.mean(results_period[j], axis=1)}")
        print(f"Oscillation period minimum gaps: {min_gaps_period}")

    get_plot(algos)
