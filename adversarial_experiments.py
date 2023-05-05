import argparse
import numpy as np
import concurrent.futures
from bai_algo_base import BAI_G_Design
from fixed_budget_rage import RAGE
from bobw_algo import P1_Linear


def single_trial(trial_id, X, T, thetas, opt_arm, algo):
    print(f"Trial {trial_id} started.")
    
    reward_func = lambda x, t: np.random.normal(x@thetas[t], 1)
    if algo == 'G_design':
        recommendation = BAI_G_Design(X, T, reward_func).run()
    elif algo == 'RAGE':
        recommendation = RAGE(X, T, reward_func).run()
    elif algo == 'Modified_RAGE':
        recommendation = RAGE(X, T, reward_func, bobw=True).run()
    elif algo == 'BOBW':
        recommendation = P1_Linear(X, T, reward_func, batch=True, alt=True, subroutine_max_iter=15).run()
    else:
        raise ValueError("Unknown algo: {}".format(algo))
    if np.all(recommendation == X[opt_arm]):
        result = 1
        print("correct! recommendation = {}".format(recommendation))
    else:
        result = 0
        print("incorrect! recommendation = {}".format(recommendation))
    print(f"Trial {trial_id} finished with result {result}.")
    return result

def run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, n_workers=None):
    if n_workers is None:
        n_workers = min(n_trials, 4) 

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(single_trial, i, X, T, thetas, opt_arm, algo): i for i in range(n_trials)}

        results = []
        for future in concurrent.futures.as_completed(futures):
            trial_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Trial {trial_id} generated an exception: {exc}")
            else:
                print(f"Trial {trial_id} completed successfully.")

    return results


def compute_gap(X, thetas):
    vals = X @ thetas.T
    ave_vals = np.mean(vals, axis=1)
    opt_arm = np.argmax(ave_vals)
    print(f'optimal arm: {opt_arm}')
    ave_vals = np.sort(ave_vals)
    gap = ave_vals[-1] - ave_vals[-2]
    print(f"Gap: {gap}")
    return gap, opt_arm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments in parallel")
    parser.add_argument("-a", "--algorithm", type=str, default='G_design',
                        help="BAI algorithm to run (default: G_design)")
    args = parser.parse_args()

    algo = args.algorithm

    d = 10
    n_trials = 5000
    omega = 0.2
    T = 30000
    move_gap = 200
    osci_mags = [0, 0.5, 1.0, 2.0, 4.0, 8.0]
    # osci_mags = [1.0]

    X = np.eye(d)
    x_extra = np.zeros(d)
    x_extra[0] = np.cos(omega)
    x_extra[1] = np.sin(omega)
    X = np.vstack((X, x_extra))

    theta1 = np.ones(d)
    theta1[0] = 0
    theta2 = np.zeros(d)
    theta2[0] = 2.0

    results_total = np.zeros((len(osci_mags), n_trials))
    ts = np.arange(T)
    for i, osci_mag in enumerate(osci_mags):
        thetas = np.zeros((T, d))
        thetas[:] = theta2
        thetas[:, -1] = osci_mag * np.sin(ts / move_gap) + 2.01
        # thetas[:int(T/3), :] = theta1
        # thetas[int(T/3)+1:, :] = theta2

        gap, opt_arm = compute_gap(X, thetas)

        results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, 6)
        results_total[i] = results

    print(f"{algo} Accuracy: {np.mean(results_total, axis=1)}")
    np.savez_compressed(f'plot_data/{algo}/{algo}_results_omega{omega}_adv3.npz', results=results_total)
