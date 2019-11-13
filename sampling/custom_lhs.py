# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import numpy as np
from sampling.lhs import LHS
import matplotlib.pyplot as plt


class SOMLHS:

	def __init__(self, n_max=None, at=None, eb=None, ds_beta=None, eps_ds=None, criterion='c'):

		if n_max is None:
			n_max = [10, 200]

		if at is None:
			at = [0.70, 0.999]

		if eb is None:
			eb = [0.001, 0.0001]

		if ds_beta is None:
			ds_beta = [2, 20]

		if eps_ds is None:
			eps_ds = [2, 20]

		self.n_max = np.array(n_max)
		self.at = np.array(at)
		self.eb = np.array(eb)
		self.ds_beta = np.array(ds_beta)
		self.eps_ds = np.array(eps_ds)

		self.limits = np.array([self.n_max, self.at, self.eb, self.ds_beta, self.eps_ds])

		self.criterion = criterion

		self.lhs = LHS(self.limits, self.criterion)

	def __call__(self, samples):
		return self.lhs(samples)


class FullModelLHS(SOMLHS):

	def __init__(self, n_max=None, at=None, eb=None, ds_beta=None, eps_ds=None, criterion='c'):
		super(SOMLHS, self).__init__(n_max, at, eb, ds_beta, eps_ds, criterion)

