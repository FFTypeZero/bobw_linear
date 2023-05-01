import numpy as np
import concurrent.futures
from bai_algo_base import BAI_G_Design
from fixed_budget_rage import RAGE
from bobw_algo import P1_Linear


omega = 0.01
d = 8
T = 30000
num_trials = 20
algo = 'G_design'

X = np.eye(d)
x_extra = np.zeros(d)
x_extra[0] = np.cos(omega)
x_extra[1] = np.sin(omega)
X = np.vstack((X, x_extra))
theta = np.zeros(d)
theta[0] = 2.0
reward_func = lambda x, t: np.random.normal(x@theta, 1)

num_correct = 0
reco_record = np.zeros(num_trials)

for _ in range(num_trials):
    print("d = {}, Trial {}/{}".format(d, _ + 1, num_trials))
    if algo == 'G_design':
        recommendation = BAI_G_Design(X, T, reward_func).run()
    elif algo == 'RAGE':
        recommendation = RAGE(X, T, reward_func).run()
    elif algo == 'BOBW':
        recommendation = P1_Linear(X, T, reward_func, batch=True, alt=True).run()
    else:
        raise ValueError("Unknown algo: {}".format(algo))
    if np.all(recommendation == X[0]):
        num_correct += 1
        reco_record[_] = 1
    else:
        print("incorrect! recommendation = {}".format(recommendation))
    print("d = {}, {} current accuracy = {}".format(d, algo, num_correct / (_ + 1)))
# np.savez_compressed('plot_data/{}/reco_record_d{}.npz'.format(algo, d), reco_record=reco_record)