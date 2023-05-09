import numpy as np
import concurrent.futures
from bai_algo_base import BAI_G_Design
from fixed_budget_rage import RAGE
from bobw_algo import P1_Linear


def single_trial(trial_id, X, T, thetas, opt_arm, algo, noise_level=1.0):
    print(f"Trial {trial_id} started.")
    
    reward_func = lambda x, t: np.random.normal(x@thetas[t], noise_level)
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
        # print("correct! recommendation = {}".format(recommendation))
    else:
        result = 0
        print("incorrect! recommendation = {}".format(recommendation))
    print(f"Trial {trial_id} finished with result {result}.")
    return result


def run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level=1.0, n_workers=None):
    if n_workers is None:
        n_workers = min(n_trials, 4) 

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(single_trial, i, X, T, thetas, opt_arm, algo, noise_level): i for i in range(n_trials)}

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
    n, d = X.shape
    thetas = thetas.reshape(-1, d)
    vals = X @ thetas.T
    ave_vals = np.mean(vals, axis=1)
    opt_arm = np.argmax(ave_vals)
    print(f'optimal arm: {opt_arm}')
    ave_vals = np.sort(ave_vals)
    gap = ave_vals[-1] - ave_vals[-2]
    print(f"Gap: {gap}")
    return gap, opt_arm
