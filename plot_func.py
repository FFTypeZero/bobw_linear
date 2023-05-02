import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    algos = ['G_design', 'RAGE', 'BOBW']
    for algo in algos:
        loaded = np.load(f'plot_data/{algo}/{algo}_results_dim.npz')
        results = loaded['results']
        ds = loaded['ds']
        accs = np.mean(results, axis=1)
        plt.plot(ds, accs, 'o-', label=algo)
    plt.xlabel('$d$')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. $d$ under Soare et al. (2014) Examples')
    plt.legend(loc='best')
    plt.show()