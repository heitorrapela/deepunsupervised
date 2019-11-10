import matplotlib.pyplot as plt

fig = ax = None
scat_samples = scat_centers = None
colors = None


def plot_data(data, target, centers=None, relevances=None):

    global fig, ax, scat_samples, scat_centers, colors

    if fig is None:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('dim0', fontsize=15)
        ax.set_ylabel('dim1', fontsize=15)
        ax.grid(True)

        colors = target*360/(target.max()+1)
        scat_samples = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap='hsv', alpha=0.5)
        scat_centers = ax.errorbar(centers[:, 0], centers[:, 1],
                                   xerr=relevances[:, 0], yerr=relevances[:, 1], alpha=0.5, fmt='o', c='k')

        fig.tight_layout()
        plt.show()

    else:
        colors = target*360/(target.max()+1)

        scat_samples.remove()
        scat_samples = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap='hsv', alpha=0.5)

        scat_centers.remove()
        scat_centers = ax.errorbar(centers[:, 0], centers[:, 1],
                                   xerr=relevances[:, 0], yerr=relevances[:, 1], alpha=0.5, fmt='o', c='k')

    plt.pause(0.01)


def plot_hold():
    plt.pause(10000)

