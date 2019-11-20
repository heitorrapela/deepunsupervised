import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np    


class Plotter:
    def __init__(self):

        self.fig = None 
        self.ax = None
        self.scat_samples = None
        self.scat_centers = None
        self.colors = None
        self.labels_samples = None

    def plot_data(self, data, target, centers=None, relevances=None, pause_time=0.01):

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlabel('dim0', fontsize=15)
            self.ax.set_ylabel('dim1', fontsize=15)
            self.ax.grid(True)
            
            colors = (2*np.pi)*target/(target.max()*2)
            self.scat_samples = self.ax.scatter(data[:, 0], data[:, 1], c=colors, cmap='hsv', alpha=0.5)
            
            self.labels_samples = []
            for i, label in enumerate(target):
                self.labels_samples.append(self.ax.annotate(label, 
                                         (data[:, 0][i], data[:, 1][i]),
                                         horizontalalignment='center',
                                         verticalalignment='center',
                                         size=11))

            if centers is not None and relevances is not None:
                self.scat_centers = self.ax.errorbar(centers[:, 0], centers[:, 1],
                                                     xerr=relevances[:, 0], yerr=relevances[:, 1],
                                                     alpha=0.5, fmt='o', c='k')

            self.fig.tight_layout()
            plt.show()

        else:
            colors = (2*np.pi)*target/(target.max()*2)
            self.scat_samples.remove()
            self.scat_samples = self.ax.scatter(data[:, 0], data[:, 1], c=colors, cmap='hsv', alpha=0.5)

            for i, label in enumerate(target):
                self.labels_samples[i].remove()
                self.labels_samples[i] = self.ax.annotate(label, 
                             (data[:, 0][i], data[:, 1][i]),
                             horizontalalignment='center',
                             verticalalignment='center',
                             size=11)
            
            if centers is not None and relevances is not None:
                self.scat_centers.remove()
                self.scat_centers = self.ax.errorbar(centers[:, 0], centers[:, 1],
                                                     xerr=relevances[:, 0], yerr=relevances[:, 1],
                                                     alpha=0.5, fmt='o', c='k')

        plt.pause(pause_time)

    def plot_hold(self, time=10000):
        plt.pause(time)
