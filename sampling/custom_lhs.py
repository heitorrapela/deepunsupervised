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
		return self.curr_sampling

	def write_params_file(self, filename):
		params = pd.DataFrame(self.curr_sampling, columns=self.params_names)
		params.to_csv(join("arguments", filename), sep=',', index=False, header=True)


class FullModelLHS(SOMLHS):

	def __init__(self, n_max=None, at=None, eb=None, ds_beta=None, eps_ds=None, epochs=None, seed=None, criterion='c'):
		super(SOMLHS, self).__init__(n_max, at, eb, ds_beta, eps_ds, epochs, seed, criterion)
