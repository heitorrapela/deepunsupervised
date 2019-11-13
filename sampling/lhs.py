# Author: Pedro Braga <phmb4@cin.ufpe.br>.

from pyDOE2 import lhs


class LHS:

    def __init__(self, limits, criterion):
        self.limits = limits
        self.criterion = criterion

    def __call__(self, n_samples):
        nx = self.limits.shape[0]
        x = lhs(nx, samples=n_samples, criterion=self.criterion)
        for kx in range(nx):
            x[:, kx] = self.limits[kx, 0] + x[:, kx] * (self.limits[kx, 1] - self.limits[kx, 0])

        return x
