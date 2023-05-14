import numpy as np
from algorithms.g_bai import BAI_Base
from algorithms.fw import fw_XY


class OD_LinBAI(BAI_Base):
    def __init__(self, X, T, reward_func) -> None:
        super().__init__(X, T, reward_func)

    def run(self):
        A_r = self.X
        A_to_X_r = [i for i in range(self.n)]
        num_epochs = np.ceil(np.log2(self.d)).astype(int)
        ds = np.zeros(num_epochs + 1, dtype=int)
        ds[0] = self.d
        numerator = np.sum(np.array([np.ceil(self.d / 2**r) for r in range(1, num_epochs)]))
        m = (self.T - np.minimum(self.n, self.d * (self.d + 1) / 2) - numerator) / num_epochs
        if m <= 0:
            print("Unenough budget! return random recommendation.")
            return self.X[np.random.randint(self.n)]

        t = 0
        for r in range(1, num_epochs + 1):
            print(f"OD-LinBAI epoch {r}/{num_epochs}")

            ds[r] = np.linalg.matrix_rank(A_r).astype(int)
            if ds[r] < ds[r - 1]:
                U_r, s_r, Vh_r = np.linalg.svd(A_r, full_matrices=False)
                A_r = A_r @ Vh_r.T[:, :ds[r]]
            pi_r = fw_XY(A_r, A_r)[0]
            allocation_r = np.ceil(pi_r * m).astype(int)

            covariance_r = np.zeros((ds[r], ds[r]))
            target_r = np.zeros(ds[r])
            samples = np.concatenate([np.repeat(j, allocation_r[j]) for j in range(len(allocation_r))])
            samples = np.random.permutation(samples)
            for j in samples:
                covariance_r += np.outer(A_r[j], A_r[j])
                target_r += A_r[j] * self.reward_func(self.X[A_to_X_r[j]], t)
                t += 1
            theta_hat_r = np.linalg.pinv(covariance_r) @ target_r
            r_hat_r = A_r @ theta_hat_r

            remain_num_r = np.ceil(self.d / 2**r).astype(int)
            sort_idx = np.flip(np.argsort(r_hat_r))
            A_r = A_r[sort_idx]
            A_r = A_r[:remain_num_r]
            A_to_X_r = [A_to_X_r[idx] for idx in sort_idx[:remain_num_r]]
            if remain_num_r <= 1:
                break

        return self.X[A_to_X_r[0]]
    

if __name__ == '__main__':
    omega = 0.1
    d = 10
    T = 20000
    num_trials = 20

    X = np.eye(d)
    x_extra = np.zeros(d)
    x_extra[0] = np.cos(omega)
    x_extra[1] = np.sin(omega)
    X = np.vstack((X, x_extra))
    theta = np.zeros(d)
    theta[0] = 2.0
    reward_func = lambda x, t: np.random.normal(x@theta, 1)

    num_correct = 0
    for _ in range(num_trials):
        print("Trial {}/{}".format(_ + 1, num_trials))
        recommendation = OD_LinBAI(X, T, reward_func).run()
        if np.all(recommendation == X[0]):
            num_correct += 1
        else:
            print("incorrect! recommendation = {}".format(recommendation))
        print("OD-LinBAI accuracy = {}".format(num_correct / (_ + 1)))
    print("OD-LinBAI accuracy = {}".format(num_correct / num_trials))