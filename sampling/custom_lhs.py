# Author: Pedro Braga <phmb4@cin.ufpe.br>.
# Author: Heitor Rapela Medeiros <hrm@cin.ufpe.br>.

import numpy as np
import pandas as pd
from sampling.lhs import LHS


class SOMLHS:

    def __init__(self, n_max=None, at=None, eb=None, ds_beta=None, eps_ds=None, ld=None,
                 epochs=None, seed=None, criterion='c'):

        if n_max is None:
            n_max = [10, 200]

        if at is None:
            at = [0.70, 0.999]

        if eb is None:
            eb = [0.0001, 0.01]

        if ds_beta is None:
            ds_beta = [0.001, 0.5]

        if eps_ds is None:
            eps_ds = [0.01, 0.1]

        if epochs is None:
            epochs = [70, 200]

        if ld is None:
            ld = [0.05, 0.5]

        if seed is None:
            seed = [1, 200000]

        self.n_max = np.array(n_max)
        self.at = np.array(at)
        self.eb = np.array(eb)
        self.ds_beta = np.array(ds_beta)
        self.eps_ds = np.array(eps_ds)
        self.ld = np.array(ld)
        self.epochs = np.array(epochs)
        self.seed = np.array(seed)

        self.limits = np.array([self.n_max, self.at, self.eb, self.ds_beta, self.eps_ds,
                                self.ld, self.epochs, self.seed])
        self.curr_sampling = []
        self.params_names = ['n_max', 'at', 'eb', 'ds_beta', 'eps_ds', 'ld', 'epochs', 'seed']

        self.criterion = criterion
        self.lhs = LHS(self.limits, self.criterion)

    def __call__(self, samples, **kwargs):
        self.curr_sampling = self.lhs(samples)
        self.curr_sampling = pd.DataFrame(self.curr_sampling, columns=self.params_names).astype('float32')

        if kwargs.get('custom_dist_n_max') is None or kwargs.get('custom_dist_n_max') == 'default':
            self.curr_sampling['n_max'] = self.curr_sampling['n_max'].round().astype('int32')
        else:
            self.curr_sampling['n_max'] = self.custom_distribution(
                self.n_max, samples, param=kwargs.get('custom_dist_n_max')).round().astype('int32')

        if kwargs.get('custom_dist_at') == 'exp' or kwargs.get('custom_dist_at') == 'exp_inv':
            self.curr_sampling['at'] = self.custom_distribution(self.at, samples,
                                                                param=kwargs.get('custom_dist_at'))

        if kwargs.get('custom_dist_eb') == 'exp' or kwargs.get('custom_dist_eb') == 'exp_inv':
            self.curr_sampling['eb'] = self.custom_distribution(self.eb, samples,
                                                                param=kwargs.get('custom_dist_eb'))

        if kwargs.get('custom_dist_ds_beta') == 'exp' or kwargs.get('custom_dist_ds_beta') == 'exp_inv':
            self.curr_sampling['ds_beta'] = self.custom_distribution(self.ds_beta, samples,
                                                                     param=kwargs.get('custom_dist_ds_beta'))

        if kwargs.get('custom_dist_eps_ds') == 'exp' or kwargs.get('custom_dist_eps_ds') == 'exp_inv':
            self.curr_sampling['eps_ds'] = self.custom_distribution(self.eps_ds, samples,
                                                                    param=kwargs.get('custom_dist_eps_ds'))

        if kwargs.get('custom_dist_ld') == 'exp' or kwargs.get('custom_dist_ld') == 'exp_inv':
            self.curr_sampling['ld'] = self.custom_distribution(self.ld, samples,
                                                                param=kwargs.get('custom_dist_ld'))

        if kwargs.get('custom_dist_epochs') is None or kwargs.get('custom_dist_epochs') == 'default':
            self.curr_sampling['epochs'] = self.curr_sampling['epochs'].round().astype('int32')
        else:
            self.curr_sampling['epochs'] = self.custom_distribution(
                self.epochs, samples, param=kwargs.get('custom_dist_epochs')).round().astype('int32')

        if kwargs.get('custom_dist_seed') is None or kwargs.get('custom_dist_seed') == 'default':
            self.curr_sampling['seed'] = self.curr_sampling['seed'].round().astype('int32')
        else:
            self.curr_sampling['seed'] = self.custom_distribution(
                self.seed, samples, param=kwargs.get('custom_dist_seed')).round().astype('int32')

        return self.curr_sampling

    def write_params_file(self, filename):
        self.curr_sampling.to_csv(filename, sep=',', index=False, header=True)

    def custom_distribution(self, limits_range, lhs_samples_n, param='default'):

        if param is None or param is 'default':
            lhs_temp = LHS(limits_range, self.criterion)
            return lhs_temp(lhs_samples_n)
        elif param == 'exp':
            limits_tmp = np.log10(np.array([limits_range]))
            lhs_temp = LHS(limits_tmp, self.criterion)
            return np.power(10, lhs_temp(lhs_samples_n))
        elif param == 'exp_inv':
            limits_tmp = np.log10(np.ones(2) - np.array([limits_range]) + np.finfo(float).eps)
            lhs_temp = LHS(limits_tmp, self.criterion)
            return 1 - np.power(10, lhs_temp(lhs_samples_n))


class FullModelLHS(SOMLHS):

    def __init__(self, lr_cnn=None, n_conv=None, som_in=None, max_pool=None, max_pool2d_size=None, filters_pow=None,
                 kernel_size=None, n_max=None, at=None, eb=None, ds_beta=None, eps_ds=None, ld=None, epochs=None,
                 seed=None, criterion='c'):
        super(FullModelLHS, self).__init__(n_max, at, eb, ds_beta, eps_ds, ld, epochs, seed, criterion)

        if lr_cnn is None:
            lr_cnn = [0.00001, 0.1]

        if n_conv is None:
            n_conv = [1, 5]

        if som_in is None:
            som_in = [2, 128]

        if max_pool is None:
            max_pool = [0, 1]

        if max_pool2d_size is None:
            max_pool2d_size = [2, 4]

        if filters_pow is None:
            filters_pow = [2, 6]

        if kernel_size is None:
            kernel_size = [0.5, 3.5]

        self.lr_cnn = np.array(lr_cnn)
        self.n_conv = np.array(n_conv)
        self.som_in = np.array(som_in)
        self.max_pool = np.array(max_pool)
        self.max_pool2d_size = np.array(max_pool2d_size)
        self.filters_pow = np.array(filters_pow)
        self.kernel_size = np.array(kernel_size)

        full_limits = np.array([self.lr_cnn, self.n_conv, self.som_in, self.max_pool,
                                self.max_pool2d_size, self.filters_pow, self.kernel_size])
        self.limits = np.concatenate((full_limits, self.limits))

        full_params_names = ['lr_cnn', 'n_conv', 'som_in', 'max_pool', 'max_pool2d_size', 'filters_pow', 'kernel_size']
        self.params_names = np.concatenate((full_params_names, self.params_names))

        self.lhs = LHS(self.limits, self.criterion)

    def __call__(self, samples, **kwargs):
        super(FullModelLHS, self).__call__(samples, **kwargs)

        if kwargs.get('custom_dist_lr_cnn') == 'exp' or kwargs.get('custom_dist_lr_cnn') == 'exp_inv':
            self.curr_sampling['lr_cnn'] = self.custom_distribution(self.lr_cnn, samples,
                                                                    param=kwargs.get('custom_dist_lr_cnn'))

        if kwargs.get('custom_dist_n_conv') is None or kwargs.get('custom_dist_n_conv') == 'default':
            self.curr_sampling['n_conv'] = self.curr_sampling['n_conv'].round().astype('int32')
        else:
            self.curr_sampling['n_conv'] = self.custom_distribution(
                self.n_conv, samples, param=kwargs.get('custom_dist_n_conv')).round().astype('int32')

        if kwargs.get('custom_dist_som_in') is None or kwargs.get('custom_dist_som_in') == 'default':
            self.curr_sampling['som_in'] = self.curr_sampling['som_in'].round().astype('int32')
        else:
            self.curr_sampling['som_in'] = self.custom_distribution(
                self.som_in, samples, param=kwargs.get('custom_dist_som_in')).round().astype('int32')

        if kwargs.get('custom_dist_max_pool') is None or kwargs.get('custom_dist_max_pool') == 'default':
            self.curr_sampling['max_pool'] = self.curr_sampling['max_pool'].round().astype('int32')
        else:
            self.curr_sampling['max_pool'] = self.custom_distribution(
                self.max_pool, samples, param=kwargs.get('custom_dist_max_pool')).round().astype('int32')

        if kwargs.get('custom_dist_max_pool2d_size') is None or kwargs.get('custom_dist_max_pool2d_size') == 'default':
            self.curr_sampling['max_pool2d_size'] = self.curr_sampling['max_pool2d_size'].round().astype('int32')
        else:
            self.curr_sampling['max_pool2d_size'] = self.custom_distribution(
                self.max_pool2d_size, samples, param=kwargs.get('custom_dist_max_pool2d_size')).round().astype('int32')

        if kwargs.get('custom_dist_filters_pow') is None or kwargs.get('custom_dist_filters_pow') == 'default':
            self.curr_sampling['filters_pow'] = self.curr_sampling['filters_pow'].round().astype('int32')
        else:
            self.curr_sampling['filters_pow'] = self.custom_distribution(
                self.filters_pow, samples, param=kwargs.get('custom_dist_filters_pow')).round().astype('int32')

        if kwargs.get('custom_dist_kernel_size') is None or kwargs.get('custom_dist_kernel_size') == 'default':
            self.curr_sampling['kernel_size'] = self.curr_sampling['kernel_size'].round().astype('int32')
        else:
            self.curr_sampling['kernel_size'] = self.custom_distribution(
                self.kernel_size, samples, param=kwargs.get('custom_dist_kernel_size')).round().astype('int32')

        # it maps the sampled values 1, 2, and 3 to the kernel_size values 3, 5 and 7, respectively
        self.curr_sampling['kernel_size'] = self.curr_sampling['kernel_size'] * 2 + 1

        return self.curr_sampling


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lhs_samples = 250
    lhs = FullModelLHS()
    dists = [None, 'exp', 'exp_inv']

    params_names = ['n_max', 'at', 'eb', 'ds_beta', 'eps_ds', 'ld', 'epochs', 'seed', 'lr_cnn', 'n_conv', 'som_in',
                    'max_pool', 'max_pool2d_size', 'filters_pow', 'kernel_size']

    # for i in range(len(params_names)):
    for dist in dists:
        sampling = lhs(lhs_samples, custom_dist_n_max=dist,
                       custom_dist_at=dist,
                       custom_dist_eb=dist,
                       custom_dist_ds_beta=dist,
                       custom_dist_eps_ds=dist,
                       custom_dist_ld=dist,
                       custom_dist_epochs=dist,
                       custom_dist_seed=dist,
                       custom_dist_lr_cnn=dist,
                       custom_dist_n_conv=dist,
                       custom_dist_som_in=dist,
                       custom_dist_max_pool=dist,
                       custom_dist_max_pool2d_size=dist,
                       custom_dist_filters_pow=dist,
                       custom_dist_kernel_size=dist)

        sampling = np.asarray(sampling)

        for i in range(len(params_names)):
            print(sampling[:, i].min(), sampling[:, i].max())
            plt.plot(sampling[:, i], np.arange(lhs_samples), '.')
            plt.title('LHS ' + str(dist) + ' ' + params_names[i])
            plt.xlim(sampling[:, i].min(), sampling[:, i].max())
            plt.pause(0.5)
            plt.clf()
