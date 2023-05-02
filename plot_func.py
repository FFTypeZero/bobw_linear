import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    algos = ['G_design', 'RAGE', 'BOBW']
    for algo in algos:
        loaded = np.load(f'plot_data/{algo}/{algo}_results_dim.npz')
        results = loaded['results']
        Ts = loaded['ds']
        accs = np.mean(results, axis=1)
        plt.plot(Ts, 1.0 - accs, 'o-', label=algo)
    plt.xlabel('$d$')
    plt.ylabel('Error Probability')
    # plt.yscale('log')
    plt.title('Error Probability vs. $d$ under Soare et al. (2014) Examples')
    plt.legend(loc='best')
    plt.show()