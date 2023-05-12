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


def add_perturbation_multi(X, theta_stars, osci_mag, move_gap, gap_plus, damped):
    T, num_features = theta_stars.shape
    theta_star = theta_stars[0]

    damping_length = T / 2
    ts = np.arange(T)
    max_mag = np.max(np.abs(theta_star))

    gap, opt_arm = compute_gap(X, theta_star)
    thetas = np.zeros((T, num_features))
    for j in range(num_features):
        move_gap = np.random.randint(move_gap, move_gap + gap_plus)
        if j // 2 == 0:
            if damped:
                thetas[:, j] = osci_mag * max_mag * np.exp(-ts / damping_length) * np.sin(2 * ts * np.pi / move_gap) + theta_star[j]
            else:
                thetas[:, j] = osci_mag * max_mag * np.sin(2 * ts * np.pi / move_gap) + theta_star[j]
        else:
            if damped:
                thetas[:, j] = osci_mag * max_mag * np.exp(-ts / damping_length) * np.cos(2 * ts * np.pi / move_gap) + theta_star[j]
            else:
                thetas[:, j] = osci_mag * max_mag * np.cos(2 * ts * np.pi / move_gap) + theta_star[j]

    gap_adv, opt_arm_adv = compute_gap(X, thetas)
    assert opt_arm_adv == opt_arm

    return thetas


def run_change_osci(algos, n_trials=1000):
    T = 10000
    noise_level = 0.3
    osci_mags = [1.0 * i for i in range(9)]

    move_gap = 200
    gap_plus = 100
    damped = False
    D = 4

    # algos = ['G_design', 'Peace', 'P1-Peace']
    results_total = np.zeros((len(algos), len(osci_mags), n_trials))
    np.random.seed(6)
    X, theta_stars = get_sto_multi(D, T)

    for i, osci_mag in enumerate(osci_mags):
        thetas = add_perturbation_multi(X, theta_stars, osci_mag, move_gap, gap_plus, damped)

        gap, opt_arm = compute_gap(X, thetas)

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, 6)
            results_total[j][i] = np.array(results)
            # np.savez_compressed(f'plot_data/{algo}/{algo}_results_multi_adv9.npz', results=results_total[j], osci_mags=osci_mags)

    for j, algo in enumerate(algos):
        print(f"{algo} Accuracy: {np.mean(results_total[j], axis=1)}")


def run_change_period(algos, n_trials=1000):
    T = 10000
    noise_level = 0.3
    osci_mag = 2.0

    move_gaps = [300 + 300 * i for i in range(1, 10)]
    gap_plus = 200
    damped = False
    D = 4

    # algos = ['G_design', 'Peace', 'P1-Peace']
    results_total = np.zeros((len(algos), len(move_gaps), n_trials))
    np.random.seed(6)
    X, theta_stars = get_sto_multi(D, T)

    for i, move_gap in enumerate(move_gaps):
        thetas = add_perturbation_multi(X, theta_stars, osci_mag, move_gap, gap_plus, damped)

        gap, opt_arm = compute_gap(X, thetas)

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, 6)
            results_total[j][i] = np.array(results)
            # np.savez_compressed(f'plot_data/{algo}/{algo}_results_multi_adv9.npz', results=results_total[j], move_gaps=move_gaps)

    for j, algo in enumerate(algos):
        print(f"{algo} Accuracy: {np.mean(results_total[j], axis=1)}")


def get_plot():
    algos = ['G-BAI', 'Peace', 'P1-Peace']
    fig, axs = plt.subplots(1, 2)
    for algo in algos:
        loaded_osci = np.load(f'plot_data/{algo}/{algo}_results_multi_adv8.npz')
        loaded_period = np.load(f'plot_data/{algo}/{algo}_results_multi_adv9.npz')
        results_osci = loaded_osci['results']
        results_period = loaded_period['results']
        osci_mags = loaded_osci['osci_mags']
        move_gaps = loaded_period['move_gaps']

        error_prob_osci = 1.0 - np.mean(results_osci, axis=1)
        error_prob_period = 1.0 - np.mean(results_period, axis=1)
        confi_bound_osci = 1.96 * np.std(results_osci, axis=1) / np.sqrt(results_osci.shape[1])
        confi_bound_period = 1.96 * np.std(results_period, axis=1) / np.sqrt(results_period.shape[1])

        axs[0].plot(osci_mags, error_prob_osci, 'o-', label=algo)
        axs[0].fill_between(osci_mags, error_prob_osci - confi_bound_osci, error_prob_osci + confi_bound_osci, alpha=0.4)
        axs[1].plot(move_gaps, error_prob_period, 'o-', label=algo)
        axs[1].fill_between(move_gaps, error_prob_period - confi_bound_period, error_prob_period + confi_bound_period, alpha=0.4)
    axs[0].set_xlabel('oscillation scale')
    axs[0].set_ylabel('error probability')
    axs[1].set_xlabel('oscillation period')
    axs[0].set_title('Error Probability vs. Oscillation Scale')
    axs[1].set_title('Error Probability vs. Oscillation Period')
    for i in range(len(axs)):
        axs[i].set_ylim([-0.05, 1.05])
        axs[i].legend(loc='best')
        axs[i].grid(True)
        axs[i].set_aspect(1.0 / axs[i].get_data_ratio(), adjustable='box')

    plt.suptitle("Experiments under Non-stationary Multivariate Testing Example")
    plt.show()


if __name__ == '__main__':
    # n_trials = 1000
    # algos = ['G-BAI', 'Peace', 'P1-Peace']
    # run_change_osci(algos, n_trials)
    # run_change_period(algos, n_trials)

    get_plot()
