import numpy as np
from fw import fw_XY


class BAI_Base:
    def __init__(self, X, T, reward_func) -> None:
        self.X = X
        self.reward_func = reward_func
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.T = T

    def __fw_XXi(self, X, X_i):
        Y_i = X_i[:, np.newaxis, :] - X_i[np.newaxis, :, :]
        design_i, rho_i = fw_XY(X, Y_i)
        return design_i, rho_i

    def run(self):
        pass


class BAI_G_Design(BAI_Base):
    def __init__(self, X, T, reward_func) -> None:
        self.G_design = fw_XY(X, X)[0]
        self.Sigma = X.T@np.diag(self.G_design)@X
        super().__init__(X, T, reward_func)

    def run(self):
        theta_hat_t = np.zeros(self.d)
        for t in range(self.T):
            i_t = np.random.choice(self.n, p=self.G_design)
            r_t = self.reward_func(self.X[i_t], t)
            theta_hat_s = np.linalg.solve(self.Sigma, self.X[i_t] * r_t)
            theta_hat_t = (theta_hat_t * t + theta_hat_s) / (t + 1)
        recommendation = np.argmax(self.X@theta_hat_t)
        return recommendation
