import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


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


class HParams:
    def __init__(self):
        self.fig = None

    def plot_hparams(self, tag, p_values, metrics):
        return self.plot_x_y(p_values, metrics, tag)

    def plot_x_y(self, x, y, title, marker="o", color='b', font_size=12):
        self.fig, ax = plt.subplots()
        ax.yaxis.grid()
        ax.set_ylim([0, 1])

        x = x.astype(float)
        y = y.astype(float)

        plt.rc('font', family='serif')
        plt.title(title, fontsize=font_size)
        plt.plot(x, y, marker, color=color, clip_on=False)
        plt.yticks(np.linspace(0, 1, num=11))

        self.plot_fit_linear(plt, x, y)

        plt.tight_layout(pad=0.2)

        return self.fig

    def plot_fit_linear(self, to_plot, x, y):
        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))

        # Make predictions using the testing set
        fit = regr.predict(x.reshape(-1, 1))

        to_plot.plot(x, fit, color='r', clip_on=False, linewidth=6)

    def plot_tensorboard_x_y(self, parameters, metric_name, metric_values, writer, dataset):
        for param, p_values in parameters.iteritems():
            if param == 'seed' or param == 'Index':
                continue

            figure = self.plot_hparams(param, p_values.values, metric_values)
            writer.add_figure(param + '/' + metric_name + '_' + dataset, figure)
            # summ_writer.add_hparams(hparam_dict=dict(param_set._asdict()),
            #                         metric_dict={'CE_' + dataset_path.split(".arff")[0]: ce})

    def check_plot_save(self, path, save, plot):
        if save:
            plt.savefig(path, bbox_inches='tight', pad_inches=0)

        if plot:
            plt.show()
        else:
            plt.close()




