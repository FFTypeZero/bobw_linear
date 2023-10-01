import os
import numpy as np
import concurrent.futures
from algorithms.g_bai import BAI_G_Design
from algorithms.peace import Peace
from algorithms.p1_linear import P1_Linear
from algorithms.od_linbai import OD_LinBAI


def single_trial(trial_id, X, T, thetas, opt_arm, algo, noise_level=1.0, setting_para='None'):
    print(f"Setting {setting_para}, Algorithm {algo}, trial {trial_id} started.")
    
    reward_func = lambda x, t: np.random.normal(x@thetas[t], noise_level)
    if algo == 'G-BAI':
        recommendation = BAI_G_Design(X, T, reward_func).run()
    elif algo == 'Peace':
        recommendation = Peace(X, T, reward_func, mixed=False).run()
    elif algo == 'Mixed-Peace':
        recommendation = Peace(X, T, reward_func, mixed=True).run()
    elif algo == 'P1-RAGE':
        recommendation = P1_Linear(X, T, reward_func, batch=True, alt=False, subroutine_max_iter=15).run()
    elif algo == 'P1-Peace':
        recommendation = P1_Linear(X, T, reward_func, batch=True, alt=True).run()
    elif algo == 'OD-LinBAI':
        recommendation = OD_LinBAI(X, T, reward_func).run()
    else:
        raise ValueError("Unknown algo: {}".format(algo))
    if np.all(recommendation == X[opt_arm]):
        result = 1
    else:
        result = 0
        print("incorrect! recommendation = {}".format(recommendation))
    print(f"Setting {setting_para}, Algorithm {algo}, trial {trial_id} finished with result {result}.")
    return result


def run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, noise_level=1.0, n_workers=None, setting_para='None'):
    if n_workers is None:
        n_workers = min(n_trials, 4) 

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(single_trial, i, X, T, thetas, opt_arm, algo, noise_level, setting_para): i for i in range(n_trials)}

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


def run_trials_in_parallel_soare(n_trials, 
                                 X, 
                                 T, 
                                 thetas, 
                                 opt_arm, 
                                 algo, 
                                 saving_dir, 
                                 saving_file, 
                                 save=True, 
                                 noise_level=1.0, 
                                 n_workers=None, 
                                 setting_para='None'):
    if n_workers is None:
        n_workers = min(n_trials, 4) 

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(single_trial, i, X, T, thetas, opt_arm, algo, noise_level, setting_para): i for i in range(n_trials)}

        results = []
        results_this_run = []
        if not os.path.exists(saving_dir) and save:
            os.makedirs(saving_dir)
        if os.path.exists(saving_file) and save:
            loaded = np.load(saving_file)
            results = list(loaded['results'])
        for future in concurrent.futures.as_completed(futures):
            trial_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                results_this_run.append(result)
                if save:
                    np.savez_compressed(saving_file, results=results, T=T)
            except Exception as exc:
                print(f"Trial {trial_id} generated an exception: {exc}")
            else:
                print(f"Trial {trial_id} completed successfully.")

    return results_this_run


def compute_gap(X, thetas):
    n, d = X.shape
    thetas = thetas.reshape(-1, d)
    vals = X @ thetas.T
    ave_vals = np.mean(vals, axis=1)
    opt_arm = np.argmax(ave_vals)
    ave_vals = np.sort(ave_vals)
    gap = ave_vals[-1] - ave_vals[-2]
    print(f"optimal arm: {opt_arm}, Gap: {gap}")
    return gap, opt_arm
