import numpy as np
from algorithms.g_bai import BAI_Base
from algorithms.fw import fw_XY


class Peace(BAI_Base):
    def __init__(self, X, T, reward_func, bobw=False, verbose=False, reg=.1) -> None:
        super().__init__(X, T, reward_func, verbose)
        self.reg = reg
        self.bobw = bobw
        self.G_design = fw_XY(X, X)[0]

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

    def __rounding(self, design, num_samples):        
        '''
        Routine to convert design to allocation over num_samples following rounding procedures in Pukelsheim.
        Code from https://github.com/jkatzsam/linear_bandits_empirical_process
        '''
        num_support = (design > 0).sum()
        support_idx = np.where(design > 0)[0]
        support = design[support_idx]
        n_round = np.ceil((num_samples - .5 * num_support) * support)
        while n_round.sum()-num_samples != 0:
            if n_round.sum() < num_samples:
                idx = np.argmin(n_round / support)
                n_round[idx] += 1
            else:
                idx = np.argmax((n_round - 1) / support)
                n_round[idx] -= 1

        allocation = np.zeros(len(design))
        allocation[support_idx] = n_round
        return allocation.astype(int)

    def __eliminate_Xi(self, threshold, X_i):
        """
        Eliminate X_i so that X_{i+1} is the largest set with rho(X_{i+1}) <= rho(X_i) / 2
        """
        rho_0 = self.__fw_XXi(self.X, X_i[:1, :])[1]
        if rho_0 > threshold:
            return X_i[:1, :]

        upper_bound = X_i.shape[0]
        lower_bound = 1
        while upper_bound - lower_bound > 1:
            mid = (upper_bound + lower_bound) // 2
            rho_mid = self.__fw_XXi(self.X, X_i[:mid, :])[1]
            if rho_mid > threshold:
                upper_bound = mid
            else:
                lower_bound = mid
        return X_i[:lower_bound, :]

    def sto_run(self):
        """
        Peace for stochastic environment
        """
        rho_start = self.__fw_XXi(self.X, self.X)[1]
        num_epoches = int(np.ceil(np.log2(rho_start)))
        epoch_length = int(np.floor(self.T / num_epoches))
        if self.verbose:
            print("Peace: num_epoches = {}, epoch_length = {}".format(num_epoches, epoch_length))

        X_i = self.X
        t = 0
        for epoch in range(num_epoches):
            # Compute design and allocation
            if self.verbose:
                print("Peace: Epoch {}/{}".format(epoch + 1, num_epoches))
            design_i, rho_i = self.__fw_XXi(self.X, X_i)
            if epoch < num_epoches - 1:
                allocation_i = self.__rounding(design_i, epoch_length)
            else:
                allocation_i = self.__rounding(design_i, self.T - epoch_length * epoch)

            # Pull arms and compute theta_hat by least square estimation
            covariance = np.zeros((self.d, self.d))
            target = np.zeros(self.d)
            samples = np.concatenate([np.repeat(j, allocation_i[j]) for j in range(len(allocation_i))])
            samples = np.random.permutation(samples)
            for j in samples:
                covariance += np.outer(self.X[j], self.X[j])
                target += self.X[j] * self.reward_func(self.X[j], t)
                t += 1
            theta_hat_i = np.linalg.pinv(covariance) @ target

            # Eliminate arms
            if epoch < num_epoches - 1:
                y_hat = X_i@theta_hat_i
                X_i = X_i[np.flip(np.argsort(y_hat))]
                X_i = self.__eliminate_Xi(rho_i / 2, X_i)
                if X_i.shape[0] == 1:
                    break

        recommendation = np.argmax(X_i@theta_hat_i)
        return X_i[recommendation]

    def bobw_run(self):
        """
        Mixed Peace for both stochastic and adversarial environments
        """
        rho_start = self.__fw_XXi(self.X, self.X)[1]
        num_epoches = int(np.ceil(np.log2(rho_start)))
        epoch_length = int(np.floor(self.T / num_epoches))
        if self.verbose:
            print("Mixed Peace: num_epoches = {}, epoch_length = {}".format(num_epoches, epoch_length))

        X_i = self.X
        theta_hat_t = np.zeros(self.d)
        t = 0
        for epoch in range(num_epoches):
            # Compute design and mix it with G_design
            if self.verbose:
                print("Modified Peace: Epoch {}/{}".format(epoch + 1, num_epoches))
            sto_design_i, rho_i = self.__fw_XXi(self.X, X_i)
            design_i = (sto_design_i + self.G_design) / 2

            # Pull arms and compute theta_hat by IPS estimation
            Sigma_i_inv = np.linalg.pinv(self.X.T @ np.diag(design_i) @ self.X)
            if epoch == num_epoches - 1:
                epoch_length = self.T - epoch_length * epoch
            for _ in range(epoch_length):
                i_t = np.random.choice(self.n, p=design_i)
                r_t = self.reward_func(self.X[i_t], t)
                target_t = self.X[i_t] * r_t

                theta_hat_s = Sigma_i_inv @ target_t
                theta_hat_t = (theta_hat_t * t + theta_hat_s) / (t + 1)
                t += 1

            # Eliminate arms for design computing
            if epoch < num_epoches - 1 and X_i.shape[0] > 1:
                y_hat = X_i @ theta_hat_t
                X_i = X_i[np.flip(np.argsort(y_hat))]
                X_i = self.__eliminate_Xi(rho_i / 2, X_i)

        recommendation = np.argmax(self.X @ theta_hat_t)
        return self.X[recommendation]

    def run(self):
        if self.bobw:
            return self.bobw_run()
        else:
            return self.sto_run()


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
        recommendation = Peace(X, T, reward_func, bobw=False).run()
        if np.all(recommendation == X[0]):
            num_correct += 1
        else:
            print("incorrect! recommendation = {}".format(recommendation))
        print("Peace current accuracy = {}".format(num_correct / (_ + 1)))
    print("Peace accuracy = {}".format(num_correct / num_trials))
