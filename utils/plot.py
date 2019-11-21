import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from matplotlib.widgets import Slider


class Plotter:
    def __init__(self, init_dim_x=0, init_dim_y=0):

        self.fig = None
        self.ax = None
        self.scat_samples = None
        self.scat_centers = None
        self.colors = None
        self.labels_samples = None
        self.input_dim = None
        self.init_dim_x = init_dim_x
        self.init_dim_y = init_dim_y
        self.dimx = None
        self.dimy = None

    def update(self, val):
        self.init_dim_x = int(self.dimx.val)
        self.init_dim_y = int(self.dimy.val)

    def press(self, event):
        if event.key == 'right':
            self.init_dim_x = self.init_dim_x + 1
        elif event.key == 'left':
            self.init_dim_x = self.init_dim_x - 1
        elif event.key == 'up':
            self.init_dim_y = self.init_dim_y + 1
        elif event.key == 'down':
            self.init_dim_y = self.init_dim_y - 1

        if self.init_dim_x < 0:
            self.init_dim_x = 0
        if self.init_dim_x >= self.input_dim:
            self.init_dim_x = self.input_dim - 1
        if self.init_dim_y < 0:
            self.init_dim_y = 0
        if self.init_dim_y >= self.input_dim:
            self.init_dim_y = self.input_dim - 1

        if event.key == 'right' or event.key == 'left':
            self.dimx.set_val(self.init_dim_x)
        elif event.key == 'up' or event.key == 'down':
            self.dimy.set_val(self.init_dim_y)

        self.fig.canvas.draw_idle()

    def plot_data(self, data, target, centers=None, relevances=None, pause_time=0.01, print_labels=False):
        self.input_dim = data.shape[-1]

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10,7))
            self.ax.set_xlabel('Dim {}'.format(self.init_dim_x), fontsize=15)
            self.ax.set_ylabel('Dim {}'.format(self.init_dim_y), fontsize=15)
            self.ax.grid(True)

            axdimx = plt.axes([0.2, 0.009, 0.65, 0.03], facecolor='lightgoldenrodyellow')
            axdimy = plt.axes([0.007, 0.17, 0.03, 0.65], facecolor='lightgoldenrodyellow')

            self.dimx = Slider(axdimx, 'Dim x', int(0), int(self.input_dim - 1), valfmt="%1.0f",
                               valinit=self.init_dim_x,
                               valstep=1, orientation='horizontal')
            self.dimy = Slider(axdimy, 'Dim y', int(0), int(self.input_dim - 1), valfmt="%1.0f",
                               valinit=self.init_dim_y,
                               valstep=1, orientation='vertical')

            self.dimx.on_changed(self.update)
            self.dimy.on_changed(self.update)
            self.fig.canvas.mpl_connect('key_press_event', self.press)

            colors = (2 * np.pi) * target / (target.max() * 2)
            self.scat_samples = self.ax.scatter(data[:, self.init_dim_x], data[:, self.init_dim_y], c=colors,
                                                cmap='hsv', alpha=0.5)

            self.labels_samples = []
            if print_labels:
                for i, label in enumerate(target):
                    self.labels_samples.append(self.ax.annotate(label,
                                                                (data[:, self.init_dim_x][i],
                                                                 data[:, self.init_dim_y][i]),
                                                                horizontalalignment='center',
                                                                verticalalignment='center',
                                                                size=11))

            if centers is not None and relevances is not None:
                self.scat_centers = self.ax.errorbar(centers[:, self.init_dim_x], centers[:, self.init_dim_y],
                                                     xerr=relevances[:, self.init_dim_x],
                                                     yerr=relevances[:, self.init_dim_y],
                                                     alpha=0.5, fmt='o', c='k')
            plt.show()
        else:
            self.ax.set_xlabel('Dim {}'.format(self.init_dim_x), fontsize=15)
            self.ax.set_ylabel('Dim {}'.format(self.init_dim_y), fontsize=15)
            colors = (2 * np.pi) * target / (target.max() * 2)
            self.scat_samples.remove()
            self.scat_samples = self.ax.scatter(data[:, self.init_dim_x], data[:, self.init_dim_y], c=colors,
                                                cmap='hsv', alpha=0.5)

            if print_labels:
                for i, label in enumerate(target):
                    self.labels_samples[i].remove()
                    self.labels_samples[i] = self.ax.annotate(label,
                                                              (
                                                                  data[:, self.init_dim_x][i],
                                                                  data[:, self.init_dim_y][i]),
                                                              horizontalalignment='center',
                                                              verticalalignment='center',
                                                              size=11)

            if centers is not None and relevances is not None:
                self.scat_centers.remove()
                self.scat_centers = self.ax.errorbar(centers[:, self.init_dim_x], centers[:, self.init_dim_y],
                                                     xerr=relevances[:, self.init_dim_x],
                                                     yerr=relevances[:, self.init_dim_y],
                                                     alpha=0.5, fmt='o', c='k')
        plt.waitforbuttonpress(timeout=pause_time)

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




