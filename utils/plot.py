import matplotlib.pyplot as plt

fig = ax = scat = None


def plot_data(data, target, centers=None, relevances=None):

    global fig, ax, scat

    if fig is None:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('dim0', fontsize=15)
        ax.set_ylabel('dim1', fontsize=15)
        colors = target*360/(target.max()+1)
        scat = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap='hsv', alpha=0.5)
        ax.grid(True)

        fig.tight_layout()
        plt.show()

    else:
        scat.remove()
        colors = target*360/(target.max()+1)
        scat = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap='hsv', alpha=0.5)

    plt.pause(0.01)


def plot_hold():
    plt.pause(10000)

