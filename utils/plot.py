import matplotlib.pyplot as plt

fig = ax = None
scat_samples = scat_centers = None
colors = None

fig1 = ax1 = None
scat_samples1 = scat_centers1 = None
colors1 = None

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
        if(centers is not None and relevances is not None):
            scat_centers = ax.errorbar(centers[:, 0], centers[:, 1],
                                    xerr=relevances[:, 0], yerr=relevances[:, 1], alpha=0.5, fmt='o', c='k')

        fig.tight_layout()
        plt.show()

    else:
        colors = target*360/(target.max()+1)

        scat_samples.remove()
        scat_samples = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap='hsv', alpha=0.5)

        
        if(centers is not None and relevances is not None):
            scat_centers.remove()
            scat_centers = ax.errorbar(centers[:, 0], centers[:, 1],
                                       xerr=relevances[:, 0], yerr=relevances[:, 1], alpha=0.5, fmt='o', c='k')

    plt.pause(0.01)


def plot_data_test(data, target, centers=None, relevances=None):

    global fig1, ax1, scat_samples1, scat_centers1, colors1

    if fig1 is None:
        plt.ion()
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel('dim0', fontsize=15)
        ax1.set_ylabel('dim1', fontsize=15)
        ax1.grid(True)

        colors1 = target*360/(target.max()+1)
        scat_samples1 = ax1.scatter(data[:, 0], data[:, 1], c=colors1, cmap='hsv', alpha=0.5)
        if(centers is not None and relevances is not None):
            scat_centers1 = ax1.errorbar(centers[:, 0], centers[:, 1],
                                    xerr=relevances[:, 0], yerr=relevances[:, 1], alpha=0.5, fmt='o', c='k')

        fig1.tight_layout()
        plt.show()

    else:
        colors = target*360/(target.max()+1)

        scat_samples1.remove()
        scat_samples1 = ax1.scatter(data[:, 0], data[:, 1], c=colors, cmap='hsv', alpha=0.5)

        
        if(centers is not None and relevances is not None):
            scat_centers1.remove()
            scat_centers1 = ax1.errorbar(centers[:, 0], centers[:, 1],
                                       xerr=relevances[:, 0], yerr=relevances[:, 1], alpha=0.5, fmt='o', c='k')

    plt.pause(0.01)

def plot_hold():
    plt.pause(10000)

