import argparse
import numpy as np
from utils import run_trials_in_parallel, compute_gap


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

def get_adv_instance_2(d, T, omega, osci_mag):
    """
    Soare et al. (2014) example with oscillating arm
    """
    move_gap1 = 200
    move_gap2 = 300
    damping_length = T

    X = np.eye(d)
    x_extra = np.zeros(d)
    x_extra[0] = np.cos(omega)
    x_extra[1] = np.sin(omega)
    X = np.vstack((X, x_extra))

    ts = np.arange(T)
    thetas = np.zeros((T, d))
    # thetas[:, 0] = 0.5 * np.exp(-ts / damping_length) * np.sin(ts / move_gap1) + 2.0
    thetas[:, 0] = 2.0
    # thetas[:, -1] = osci_mag * np.exp(-ts / damping_length) * np.cos(ts * np.pi / move_gap2) + 1.95
    thetas[:, -1] = osci_mag * np.sin(2 * ts * np.pi / move_gap2) + 2.05

    return X, thetas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments in parallel")
    parser.add_argument("-a", "--algorithm", type=str, default='G_design',
                        help="BAI algorithm to run (default: G_design)")
    args = parser.parse_args()

    algo = args.algorithm

    d = 10
    omega = 0.3
    T = 30000
    move_gap1 = 200
    move_gap2 = 300
    damping_length = T
    # osci_mags = [0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    osci_mags = [2.0]

    n_trials = 20
    results_total = np.zeros((len(osci_mags), n_trials))

    for i, osci_mag in enumerate(osci_mags):
        X, thetas = get_adv_instance_2(d, T, omega, osci_mag)

        gap, opt_arm = compute_gap(X, thetas)

        results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, 6)
        results_total[i] = results
        # np.savez_compressed(f'plot_data/{algo}/{algo}_results_omega{omega}_adv4.npz', results=results_total)

    print(f"{algo} Accuracy: {np.mean(results_total, axis=1)}")
