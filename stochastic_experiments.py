import argparse
import itertools
import numpy as np
from utils import run_trials_in_parallel, compute_gap


def get_sto_instance_1(d, T, omega):
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


def get_sto_instance_2(D, T, num_sparse=None):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments in parallel")
    parser.add_argument("-a", "--algorithm", type=str, default='G_design',
                        help="BAI algorithm to run (default: G_design)")
    args = parser.parse_args()

    # algo = args.algorithm
    # ds = [8, 9, 10, 11, 12, 13]
    d = 10
    omega = 0.1
    # Ts = np.array([10 + 1000 * i for i in range(11)])
    T = 10000

    # Ds = [3, 4, 5, 6]
    Ds = [3]

    n_trials = 50
    results_total = [np.zeros((len(Ds), n_trials)), np.zeros((len(Ds), n_trials)), np.zeros((len(Ds), n_trials))]
    algos = ['G_design', 'RAGE', 'BOBW']

    for i, D in enumerate(Ds):
        # X, thetas = get_sto_instance_1(d, T, omega)
        X, thetas = get_sto_instance_2(D, T)

        gap, opt_arm = compute_gap(X, thetas)

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, 6)
            results_total[j][i] = np.array(results)
            # np.savez_compressed(f'plot_data/{algo}/{algo}_results_omega{omega}_budget2.npz', results=results_total[j], Ts=Ts)
        # np.savez_compressed(f'plot_data/{algo}/{algo}_results_omega{omega}_budget2.npz', results=results_total, Ts=Ts)

    for j, algo in enumerate(algos):
        print(f"{algo} Accuracy: {np.mean(results_total[j], axis=1)}")
    # print(f"{algo} Accuracy: {np.mean(results_total, axis=1)}")
