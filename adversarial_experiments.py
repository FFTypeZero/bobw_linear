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
        recommendation = P1_Linear(X, T, reward_func, batch=True, alt=True).run()
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
    print(f'optimal arm: {np.argmax(ave_vals)}')
    ave_vals = np.sort(ave_vals)
    gap = ave_vals[-1] - ave_vals[-2]
    return gap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments in parallel")
    parser.add_argument("-a", "--algorithm", type=str, default='G_design',
                        help="BAI algorithm to run (default: G_design)")
    args = parser.parse_args()

    algo = args.algorithm

    d = 10
    n_trials = 5000
    omega = 0.5
    T = 30000
    initial_stay = 10
    move_gap = 200

    X = np.eye(d)
    x_extra = np.zeros(d)
    x_extra[0] = np.cos(omega)
    x_extra[1] = np.sin(omega)
    X = np.vstack((X, x_extra))

    theta1 = np.ones(d)
    theta1[0] = 0
    theta2 = np.zeros(d)
    theta2[0] = 2.0

    # ts = [i for i in range(move_gap)]
    # stay_count = 1
    # while len(ts) < int(T/3):
    #     stay_length = int(initial_stay * stay_count)
    #     ts.extend([i for i in range(move_gap, 0, -1)])
    #     ts.extend([0 for j in range(stay_length)])
    #     ts.extend([i for i in range(move_gap)])
    #     stay_count += 1
    # stay_count = 1
    # while len(ts) < T:
    #     stay_length = int(10 * initial_stay * stay_count)
    #     ts.extend([move_gap for j in range(stay_length)])
    #     ts.extend([i for i in range(move_gap, 0, -1)])
    #     ts.extend([i for i in range(move_gap)])
    #     stay_count += 1
    # ts = np.array(ts[:T])

    thetas = np.zeros((T, d))
    thetas[:int(T/3), :] = theta1
    thetas[int(T/3)+1:, :] = theta2
    # thetas[:, 0] = 2.0 * np.cos(ts * omega / move_gap)
    # thetas[:, 1] = 2.0 * np.sin(ts * omega / move_gap)

    # gap = compute_gap(X, thetas)
    # print(f"Gap: {gap}")

    opt_arm = 0
    results = run_trials_in_parallel(n_trials, X, T, thetas, opt_arm, algo, 6)

    print(f"{algo} Accuracy: {np.mean(results)}")
    np.savez_compressed(f'plot_data/{algo}/{algo}_results_omega{omega}_adv.npz', results=results, thetas=thetas)
