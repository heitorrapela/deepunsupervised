# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import numpy as np
import pandas as pd
from os.path import join
from sampling.lhs import LHS


class SOMLHS:

	def __init__(self, n_max=None, at=None, eb=None, ds_beta=None, eps_ds=None, epochs=None, seed=None, criterion='c'):

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

		if seed is None:
			seed = [1, 200000]

		self.n_max = np.array(n_max)
		self.at = np.array(at)
		self.eb = np.array(eb)
		self.ds_beta = np.array(ds_beta)
		self.eps_ds = np.array(eps_ds)
		self.epochs = np.array(epochs)
		self.seed = np.array(seed)

		self.limits = np.array([self.n_max, self.at, self.eb, self.ds_beta, self.eps_ds, self.epochs, self.seed])
		self.curr_sampling = []
		self.params_names = ['n_max', 'at', 'eb', 'ds_beta', 'eps_ds', 'epochs', 'seed']

		self.criterion = criterion

		self.lhs = LHS(self.limits, self.criterion)

	def __call__(self, samples):
		self.curr_sampling = self.lhs(samples)
		self.curr_sampling = pd.DataFrame(self.curr_sampling, columns=self.params_names).astype('float32')

		self.curr_sampling['n_max'] = self.curr_sampling['n_max'].round().astype('int')
		self.curr_sampling['epochs'] = self.curr_sampling['epochs'].round().astype('int')
		self.curr_sampling['seed'] = self.curr_sampling['seed'].round().astype('int')

		return self.curr_sampling

	def write_params_file(self, filename):
		self.curr_sampling.to_csv(filename, sep=',', index=False, header=True)


class FullModelLHS (SOMLHS):

	def __init__(self, n_conv=None, som_in=None, max_pool=None, max_pool2d_size=None, filters_pow=None,
				 kernel_size=None, n_max=None, at=None, eb=None, ds_beta=None, eps_ds=None, epochs=None,
				 seed=None, criterion='c'):
		super(FullModelLHS, self).__init__(n_max, at, eb, ds_beta, eps_ds, epochs, seed, criterion)

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

		self.n_conv = np.array(n_conv)
		self.som_in = np.array(som_in)
		self.max_pool = np.array(max_pool)
		self.max_pool2d_size = np.array(max_pool2d_size)
		self.filters_pow = np.array(filters_pow)
		self.kernel_size = np.array(kernel_size)

		full_limits = np.array([self.n_conv, self.som_in, self.max_pool,
								self.max_pool2d_size, self.filters_pow, self.kernel_size])
		self.limits = np.concatenate((full_limits, self.limits))

		full_params_names = ['n_conv', 'som_in', 'max_pool', 'max_pool2d_size', 'filters_pow', 'kernel_size']
		self.params_names = np.concatenate((full_params_names, self.params_names))

		self.lhs = LHS(self.limits, self.criterion)

	def __call__(self, samples):
		super(FullModelLHS, self).__call__(samples)

		self.curr_sampling['n_conv'] = self.curr_sampling['n_conv'].round().astype('int32')
		self.curr_sampling['som_in'] = self.curr_sampling['som_in'].round().astype('int32')
		self.curr_sampling['max_pool'] = self.curr_sampling['max_pool'].round().astype('int32')
		self.curr_sampling['max_pool2d_size'] = self.curr_sampling['max_pool2d_size'].round().astype('int32')
		self.curr_sampling['filters_pow'] = self.curr_sampling['filters_pow'].round().astype('int32')
		self.curr_sampling['kernel_size'] = self.curr_sampling['kernel_size'].round().astype('int32')

		# it maps the sampled values 1, 2, and 3 to the kernel_size values 3, 5 and 7, respectively
		self.curr_sampling['kernel_size'] = self.curr_sampling['kernel_size'] * 2 + 1

		return self.curr_sampling











