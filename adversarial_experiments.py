import argparse
import itertools
import numpy as np
from utils import run_trials_in_parallel, compute_gap
from stochastic_experiments import get_sto_instance_2


def get_adv_instance_1(d, T, omega):
    """
    Adversarial example to fail RAGE
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

def get_adv_instance_2(d, T, omega, osci_mag, move_gap):
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
    # thetas[:, 0] = 0.5 * np.exp(-ts / damping_length) * np.sin(ts / move_gap1) + 2.0
    thetas[:, 0] = 0.3
    # thetas[:, -1] = osci_mag * np.exp(-ts / damping_length) * np.cos(ts * np.pi / move_gap2) + 1.95
    thetas[:, -1] = - osci_mag * np.sin(2 * ts * np.pi / move_gap) + 0.5

    return X, thetas


def add_perturbation_3(X, theta_stars, osci_mag, damped):
    T, num_features = theta_stars.shape
    theta_star = theta_stars[0]

    damping_length = T / 2
    ts = np.arange(T)
    max_mag = np.max(np.abs(theta_star))

    gap, opt_arm = compute_gap(X, theta_star)
    thetas = np.zeros((T, num_features))
    for j in range(num_features):
        move_gap = np.random.randint(300, 500)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments in parallel")
    parser.add_argument("-a", "--algorithm", type=str, default='G_design',
                        help="BAI algorithm to run (default: G_design)")
    args = parser.parse_args()

    algo = args.algorithm

    d = 10
    omega = 0.5
    T = 10000
    noise_level = 0.3
    # osci_mags = [2.0 * i for i in range(11)]
    osci_mags = [1.0]
    move_gaps = [400 + 400 * i for i in range(1, 10)]
    # move_gaps = [2400]
    num_settings = len(osci_mags)

    damped = False
    D = 4

    n_trials = 20
    # results_total = np.zeros((len(osci_mags), n_trials))
    results_total = [np.zeros((num_settings, n_trials)), np.zeros((num_settings, n_trials)), np.zeros((num_settings, n_trials))]
    algos = ['G_design', 'RAGE', 'BOBW']
    np.random.seed(6)
    X, theta_stars = get_sto_instance_2(D, T)

    for i, osci_mag in enumerate(osci_mags):
        # X, thetas = get_adv_instance_2(d, T, omega, osci_mag, move_gap)
        thetas = add_perturbation_3(X, theta_stars, osci_mag, damped)

        gap, opt_arm = compute_gap(X, thetas)

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, 6)
            results_total[j][i] = np.array(results)
            # np.savez_compressed(f'plot_data/{algo}/{algo}_results_multi_adv8.npz', results=results_total[j], osci_mags=osci_mags)
        # results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, 6)
        # results_total[i] = np.array(results)
        # np.savez_compressed(f'plot_data/{algo}/{algo}_results_multi_adv5.npz', results=results_total, osci_mags=osci_mags)

    for j, algo in enumerate(algos):
        print(f"{algo} Accuracy: {np.mean(results_total[j], axis=1)}")
    # print(f"{algo} Accuracy: {np.mean(results_total, axis=1)}")
