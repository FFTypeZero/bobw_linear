import numpy as np
from algorithms.fw import fw_XY


class BAI_Base:
    def __init__(self, X, T, reward_func) -> None:
        self.X = X
        self.reward_func = reward_func
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.T = T

    def run(self):
        pass


class BAI_G_Design(BAI_Base):
    def __init__(self, X, T, reward_func) -> None:
        self.G_design = fw_XY(X, X)[0]
        print("G_design computation complete.")
        self.Sigma = X.T@np.diag(self.G_design)@X
        super().__init__(X, T, reward_func)

    def run(self):
        theta_hat_t = np.zeros(self.d)
        Sigma_inv = np.linalg.pinv(self.Sigma)
        for t in range(self.T):
            i_t = np.random.choice(self.n, p=self.G_design)
            r_t = self.reward_func(self.X[i_t], t)
            target_t = self.X[i_t] * r_t
            theta_hat_s = Sigma_inv@target_t
            theta_hat_t = (theta_hat_t * t + theta_hat_s) / (t + 1)
            if t % 5000 == 0:
                print("G-BAI: t = {}".format(t))
        recommendation = np.argmax(self.X@theta_hat_t)
        return self.X[recommendation]

if __name__ == '__main__':
    omega = 0.01
    d = 10
    T = 30000
    num_trials = 2000

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
        recommendation = BAI_G_Design(X, T, reward_func).run()
        if np.all(recommendation == X[0]):
            num_correct += 1
        else:
            print("incorrect! recommendation = {}".format(recommendation))
        print("G-BAI current accuracy = {}".format(num_correct / (_ + 1)))
    print("B-BAI accuracy = {}".format(num_correct / num_trials))
