import numpy as np
import matplotlib.pyplot as plt
from utils import run_trials_in_parallel, compute_gap


def get_adv_Soare_1(d, T, omega, osci_mag, move_gap=200):
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
    thetas[:, 0] = 2.0
    thetas[:, -1] = osci_mag * np.sin(2 * ts * np.pi / move_gap) + 2.25

    return X, thetas


def run_change_osci(algos, n_trials=1000):
    d = 10
    omega = 0.5
    T = 10000
    noise_level = 1.0
    osci_mags = [2.0 * i for i in range(9)]

    # algos = ['G_design', 'Peace', 'P1-Peace']
    results_total = np.zeros((len(algos), len(osci_mags), n_trials))
    np.random.seed(6)

    for i, osci_mag in enumerate(osci_mags):
        X, thetas = get_adv_Soare_1(d, T, omega, osci_mag)

        gap, opt_arm = compute_gap(X, thetas)

        for j, algo in enumerate(algos):
            results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level, 6)
            results_total[j][i] = np.array(results)
            # np.savez_compressed(f'plot_data/{algo}/{algo}_results_multi_adv9.npz', results=results_total[j], move_gaps=move_gaps)

    for j, algo in enumerate(algos):
        print(f"{algo} Accuracy: {np.mean(results_total[j], axis=1)}")


def get_adv_Soare_2(d, T, omega, move_gap, osci_mag=1.0):
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

    return X, thetas

def run_change_period(algos, n_trials=1000):
    d = 10
    omega = 0.5
    T = 10000
    noise_level = 1.0
    move_gaps = [400 + 400 * i for i in range(1, 10)]

    # algos = ['G_design', 'Peace', 'P1-Peace']
    results_total = np.zeros((len(algos), len(move_gaps), n_trials))
    np.random.seed(6)

    for i, move_gap in enumerate(move_gaps):
        X, thetas = get_adv_Soare_2(d, T, omega, move_gap)

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
        loaded_osci = np.load(f'plot_data/{algo}/{algo}_results_omega0.5_adv6.npz')
        loaded_period = np.load(f'plot_data/{algo}/{algo}_results_omega0.5_adv7.npz')
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
    axs[0].set_xlabel('oscillation magnitude')
    axs[0].set_ylabel('error probability')
    axs[1].set_xlabel('oscillation period')
    axs[0].set_title('Error Probability vs. Oscillation Magnitude')
    axs[1].set_title('Error Probability vs. Oscillation Period')
    for i in range(len(axs)):
        axs[i].set_ylim([-0.05, 1.05])
        axs[i].legend(loc='best')
        axs[i].grid(True)
        axs[i].set_aspect(1.0 / axs[i].get_data_ratio(), adjustable='box')
    plt.suptitle("Experiments under Non-stationary Soare et al. (2014)'s Example")
    plt.show()


if __name__ == '__main__':
    # n_trials = 1000
    # algos = ['G-BAI', 'Peace', 'P1-Peace']
    # run_change_osci(algos, n_trials)
    # run_change_period(algos, n_trials)

    get_plot()
