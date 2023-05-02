import argparse
import numpy as np
import concurrent.futures
from bai_algo_base import BAI_G_Design
from fixed_budget_rage import RAGE
from bobw_algo import P1_Linear


def single_trial(trial_id, X, T, theta, algo):
    print(f"Trial {trial_id} started.")
    
    reward_func = lambda x, t: np.random.normal(x@theta, 1)
    if algo == 'G_design':
        recommendation = BAI_G_Design(X, T, reward_func).run()
    elif algo == 'RAGE':
        recommendation = RAGE(X, T, reward_func).run()
    elif algo == 'Modified_RAGE':
        recommendation = RAGE(X, T, reward_func, bobw=True).run()
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

def run_trials_in_parallel(n_trials, X, T, theta, algo, n_workers=None):
    if n_workers is None:
        n_workers = min(n_trials, 4) 

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(single_trial, i, X, T, theta, algo): i for i in range(n_trials)}

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
    # parser.add_argument("-d", "--dimension", type=int, default=10,
    #                     help="Arm dimensions (default: 8)")
    parser.add_argument("-a", "--algorithm", type=str, default='G_design',
                        help="BAI algorithm to run (default: G_design)")
    args = parser.parse_args()

    algo = args.algorithm
    ds = [8, 9, 10, 11, 12, 13]
    # d = 10
    n_trials = 1500
    omega = 0.01
    # Ts = np.array([12000 + 3000 * i for i in range(7)])
    T = 30000
    results_total = np.zeros((len(ds), n_trials))

    for i, d in enumerate(ds):
        X = np.eye(d)
        x_extra = np.zeros(d)
        x_extra[0] = np.cos(omega)
        x_extra[1] = np.sin(omega)
        X = np.vstack((X, x_extra))
        theta = np.zeros(d)
        theta[0] = 2.0

        results = run_trials_in_parallel(n_trials, X, T, theta, algo, 6)
        results_total[i] = np.array(results)

    np.savez_compressed(f'plot_data/{algo}/{algo}_results_dim.npz', results=results_total, ds=ds)
    print(f"{algo} Accuracy: {np.mean(results_total, axis=1)}")
