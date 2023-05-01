import argparse
import numpy as np
import concurrent.futures
from bai_algo_base import BAI_G_Design
from fixed_budget_rage import RAGE
from bobw_algo import P1_Linear


def single_trial(trial_id, d, algo):
    print(f"Trial {trial_id} started.")
    omega = 0.01
    T = 30000

    X = np.eye(d)
    x_extra = np.zeros(d)
    x_extra[0] = np.cos(omega)
    x_extra[1] = np.sin(omega)
    X = np.vstack((X, x_extra))
    theta = np.zeros(d)
    theta[0] = 2.0
    reward_func = lambda x, t: np.random.normal(x@theta, 1)
    if algo == 'G_design':
        recommendation = BAI_G_Design(X, T, reward_func).run()
    elif algo == 'RAGE':
        recommendation = RAGE(X, T, reward_func).run()
    elif algo == 'BOBW':
        recommendation = P1_Linear(X, T, reward_func, batch=True, alt=True).run()
    else:
        raise ValueError("Unknown algo: {}".format(algo))
    if np.all(recommendation == X[0]):
        result = 1
        print("correct! recommendation = {}".format(recommendation))
    else:
        result = 0
        print("incorrect! recommendation = {}".format(recommendation))
    print(f"Trial {trial_id} finished with result {result}.")
    return result

def run_trials_in_parallel(n_trials, d, algo, n_workers=None):
    if n_workers is None:
        n_workers = min(n_trials, 4)  # Use at most 4 workers by default

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all the trials to the executor
        futures = {executor.submit(single_trial, i, d, algo): i for i in range(n_trials)}

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments in parallel")
    parser.add_argument("-d", "--dimension", type=int, default=8,
                        help="Arm dimensions (default: 8)")
    parser.add_argument("-a", "--algorithm", type=str, default='G_design',
                        help="BAI algorithm to run (default: G_design)")
    args = parser.parse_args()

    d = args.dimension
    algo = args.algorithm
    n_trials = 1500
    results = run_trials_in_parallel(n_trials, d, algo, 6)
    results = np.array(results)

    print(f"{algo} Accuracy: {np.mean(results)}")
    np.savez_compressed('plot_data/{}/{}_results_d{}.npz'.format(algo, algo, d), results=results)
