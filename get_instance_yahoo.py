import numpy as np


def yahoo_problem_instance(medium_arms, best_arm, X_final, num_x, dim):
    assert num_x >= dim
    while True:
        medium_arm_set = np.random.choice(medium_arms, num_x - 1, replace=False).tolist()
        arm_set = [best_arm] + medium_arm_set
        X_subset = X_final[arm_set]
        set_rank = np.linalg.matrix_rank(X_subset)
        print(f"X rank: {set_rank}")
        if set_rank == dim:
            break
    return X_subset


if __name__ == '__main__':
    num_x = 24
    days = [f'0{i + 1}' for i in range(7)]
    thetas = []

    for day in days:
        X = np.load(f"yahoo_features_pca/yahoo_features_{day}_pca.npy")
        Y = np.load(f"yahoo_targets/yahoo_targets_{day}.npy")

        n, d = X.shape

        theta_star = np.linalg.inv(X.T@X + .01*np.eye(d))@X.T@Y
        thetas.append(theta_star)

    thetas = np.array(thetas)
    X = np.load(f"yahoo_features_pca/yahoo_features_01_pca.npy")
    theta_bar = np.mean(thetas, axis=0)

    rewards = X@theta_bar
    best_arm = np.argmax(rewards)
    max_reward = np.max(rewards)
    medium_arms = np.where(np.logical_and(rewards < max_reward - 0.05, rewards > max_reward - 1.0))[0]
    X_set = yahoo_problem_instance(medium_arms, best_arm, X, num_x, d)

    set_rewards = X_set@theta_bar
    set_gaps = np.max(set_rewards) - set_rewards
    print(f"set gaps: {np.sort(set_gaps)}")

    np.savez_compressed('yahoo_data_pca.npz', thetas=thetas, Xs=X_set)
