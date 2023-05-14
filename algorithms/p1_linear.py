import numpy as np
import matplotlib.pyplot as plt
from algorithms.g_bai import BAI_Base
from algorithms.fw import fw_XY


class P1_Linear(BAI_Base):
    def __init__(self, X, T, reward_func, batch=False, alt=False, subroutine_max_iter=15) -> None:
        self.G_design = fw_XY(X, X)[0]
        self.max_iter = subroutine_max_iter
        self.alt = alt
        self.batch = batch

        if alt:
            self.n = X.shape[0]
            self.rho_vals = np.zeros(self.n)
            self.design_vals = np.zeros((self.n, self.n))
            self.computed = np.zeros(self.n)
        super().__init__(X, T, reward_func)

    def __fw_XXi(self, X, X_i):
        N_i = X_i.shape[0]
        if N_i == 1:
            return np.ones(self.n) / self.n, 0

        Y_i_temp = X_i[:, np.newaxis, :] - X_i[np.newaxis, :, :]
        Y_i_all = np.reshape(Y_i_temp, (-1, self.d))
        rows, cols = np.triu_indices(N_i, k=1)
        idx_i = np.ravel_multi_index((rows, cols), (N_i, N_i))
        Y_i = Y_i_all[idx_i]
        design_i, rho_i = fw_XY(X, Y_i)
        return design_i, rho_i

    def rage_elimination(self, theta_hat):
        # Sort arms based on theta_hat
        y_hat = self.X@theta_hat
        sorted_idx = np.flip(np.argsort(y_hat))
        X_i = self.X[sorted_idx]

        design_bar = np.zeros(self.n)
        i_count = 0
        n_i = self.n - 1
        while n_i > 0 and i_count < self.max_iter:
            design_i = self.__fw_XXi(self.X, X_i)[0]
            design_bar += design_i
            i_count += 1

            # Virtually eliminate arms with esimated gaps larger than 2^i
            Delta_hat_i = X_i[0]@theta_hat - X_i@theta_hat
            X_i = X_i[Delta_hat_i <= 2**(-i_count)]
            n_i = X_i.shape[0] - 1

        design_out = design_bar / (2 * i_count) + self.G_design / 2
        print("P1_Linear: RAGE_Elimination rounds = {}".format(i_count))
        return design_out

    def __eliminate_Xi(self, threshold, X_i):
        """
        Eliminate X_i so that X_{i+1} is the largest set with rho(X_{i+1}) <= rho(X_i) / 2
        """
        if self.computed[0]:
            rho_0 = self.rho_vals[0]
        else:
            design_0, rho_0 = self.__fw_XXi(self.X, X_i[:1, :])
            self.design_vals[0] = design_0
            self.rho_vals[0] = rho_0
            self.computed[0] = True
        if rho_0 > threshold:
            return X_i[:1, :]

        upper_bound = X_i.shape[0]
        lower_bound = 1
        while upper_bound - lower_bound > 1:
            mid = (upper_bound + lower_bound) // 2
            if self.computed[mid - 1]:
                rho_mid = self.rho_vals[mid - 1]
            else:
                design_mid, rho_mid = self.__fw_XXi(self.X, X_i[:mid, :])
                self.design_vals[mid - 1] = design_mid
                self.rho_vals[mid - 1] = rho_mid
                self.computed[mid - 1] = True
            if rho_mid > threshold:
                upper_bound = mid
            else:
                lower_bound = mid
        return X_i[:lower_bound, :]

    def peace_elimination(self, theta_hat):
        # Sort arms based on theta_hat
        y_hat = self.X@theta_hat
        sorted_idx = np.flip(np.argsort(y_hat))
        X_i = self.X[sorted_idx]

        self.rho_vals = np.zeros(self.n)
        self.design_vals = np.zeros((self.n, self.n))
        self.computed = np.zeros(self.n)

        design_bar = np.zeros(self.n)
        i_count = 0
        n_i = self.n
        while n_i > 1:
            # Compute design of current X_i
            if self.computed[n_i - 1]:
                design_i = self.design_vals[n_i - 1]
                rho_i = self.rho_vals[n_i - 1]
            else:
                design_i, rho_i = self.__fw_XXi(self.X, X_i)
                self.design_vals[n_i - 1] = design_i
                self.rho_vals[n_i - 1] = rho_i
                self.computed[n_i - 1] = True
            design_bar += design_i
            i_count += 1

            # Virtually eliminate arms so that the remaining arms satisfy rho(X_{i+1}) <= rho(X_i) / 2
            X_i = self.__eliminate_Xi(rho_i / 2, X_i)
            n_i = X_i.shape[0]

        design_out = design_bar / (2 * i_count) + self.G_design / 2
        print("P1_Linear: Peace_Elimination rounds = {}".format(i_count))
        return design_out

    def run(self):
        if self.batch:
            rho_start = self.__fw_XXi(self.X, self.X)[1]
            num_epoches = int(np.ceil(np.log2(rho_start)))
            epoch_length = int(np.floor(self.T / num_epoches))

        design_t = self.G_design
        theta_hat_t = np.zeros(self.d)
        Sigma_t_inv = np.linalg.pinv(self.X.T@np.diag(design_t)@self.X)
        for t in range(self.T):
            # Pull arms and compute theta_hat by IPS estimation
            i_t = np.random.choice(self.n, p=design_t)
            r_t = self.reward_func(self.X[i_t], t)
            target_t = self.X[i_t] * r_t
            theta_hat_s = Sigma_t_inv@target_t
            theta_hat_t = (theta_hat_t * t + theta_hat_s) / (t + 1)

            # Compute design for next round
            if self.batch and t % epoch_length != 0:
                continue
            if self.alt:
                print(f"P1_Linear: starts round {t} Peace_Elimination.")
                design_t = self.peace_elimination(theta_hat_t)
                Sigma_t_inv = np.linalg.pinv(self.X.T@np.diag(design_t)@self.X)
            else:
                print(f"P1_Linear: starts round {t} RAGE_Elimination.")
                design_t = self.rage_elimination(theta_hat_t)
                Sigma_t_inv = np.linalg.pinv(self.X.T@np.diag(design_t)@self.X)
        recommendation = np.argmax(self.X@theta_hat_t)
        return self.X[recommendation]

if __name__ == '__main__':
    omega = 0.1
    d = 10
    T = 30000
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
        recommendation = P1_Linear(X, T, reward_func, batch=True, alt=True).run()
        if np.all(recommendation == X[0]):
            num_correct += 1
        else:
            print("incorrect! recommendation = {}".format(recommendation))
        print("P1-Linear current accuracy = {}".format(num_correct / (_ + 1)))
    print("P1-Linear accuracy = {}".format(num_correct / num_trials))
