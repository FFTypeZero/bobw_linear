import numpy as np
import matplotlib.pyplot as plt
from bai_algo_base import BAI_Base
from fw import fw_XY


class P1_Linear(BAI_Base):
    def __init__(self, X, T, reward_func, subroutine_max_iter=100) -> None:
        self.G_design = fw_XY(X, X)[0]
        self.max_iter = subroutine_max_iter
        super().__init__(X, T, reward_func)

    def subroutine(self, theta_hat):
        y_hat = self.X@theta_hat
        sorted_idx = np.argsort(y_hat, order='descending')
        X_i = self.X[sorted_idx]
        Delta_hat = X_i[0]@theta_hat - y_hat[sorted_idx]
        Y_i = X_i[:, np.newaxis, :] - X_i[np.newaxis, :, :]

        design_bar = np.zeros(self.n)
        i_count = 1
        n_i = self.n
        while n_i > 1 and i_count < self.max_iter:
            design_i = fw_XY(X_i, Y_i)[0]
            design_bar += np.concatenate(design_i, np.zeros(self.n - n_i))
            X_i = X_i[Delta_hat <= 2**(i_count)]
            Y_i = X_i[:, np.newaxis, :] - X_i[np.newaxis, :, :]
            n_i = X_i.shape[0]
            i_count += 1
        design_bar = design_bar[np.argsort(sorted_idx)]
        design_out = design_bar / (2 * i_count) + self.G_design / 2
        return design_out

    def run(self):
        design_t = self.G_design
        theta_hat_t = np.zeros(self.d)
        for t in range(self.T):
            i_t = np.random.choice(self.n, p=design_t)
            r_t = self.reward_func(self.X[i_t], t)
            Sigma_t = self.X.T@np.diag(design_t)@self.X
            theta_hat_s = np.linalg.solve(Sigma_t, self.X[i_t] * r_t)
            theta_hat_t = (theta_hat_t * t + theta_hat_s) / (t + 1)
            design_t = self.subroutine(theta_hat_t)
        recommendation = np.argmax(self.X@theta_hat_t)
        return recommendation
