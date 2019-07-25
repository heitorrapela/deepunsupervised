import torch
import torch.nn as nn
from torchvision.utils import save_image
from pyDOE import *
import math

class SOM(nn.Module):
<<<<<<< HEAD
    def __init__(self, input_size, out_size=(10, 10), lr=0.3, sigma=None):
        '''

        :param input_size:
        :param out_size:
        :param lr:
        :param sigma:
        '''
        super(SOM, self).__init__()
        self.input_size = input_size
        self.out_size = out_size

        self.lr = lr
        if sigma is None:
            self.sigma = max(out_size) / 2
        else:
            self.sigma = float(sigma)

        self.weight = nn.Parameter(torch.randn(input_size, out_size[0] * out_size[1]), requires_grad=False)
        # self.locations = nn.Parameter(torch.Tensor(list(self.get_map_index())), requires_grad=False)
        self.pdist_fn = nn.PairwiseDistance(p=2)

    def get_map_index(self):
        '''Two-dimensional mapping function'''
        for x in range(self.out_size[0]):
            for y in range(self.out_size[1]):
                yield (x, y)

    def _neighborhood_fn(self, input, current_sigma):
        '''e^(-(input / sigma^2))'''
        input.div_(current_sigma ** 2)
        input.neg_()
        input.exp_()

        return input

    def forward(self, input):
        '''
        Find the location of best matching unit.
        :param input: data
        :return: location of best matching unit, loss
        '''
        batch_size = input.size()[0]
        input = input.view(batch_size, -1, 1)
        batch_weight = self.weight.expand(batch_size, -1, -1)

        dists = self.pdist_fn(input, batch_weight)
        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)
        # bmu_locations = self.locations[bmu_indexes]

        return losses.sum().div_(batch_size).item()

    def self_organizing(self, input, current_iter, max_iter):
        '''
        Train the Self Oranizing Map(SOM)
        :param input: training data
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss (minimum distance)
        '''
        batch_size = input.size()[0]
        #Set learning rate
        iter_correction = 1.0 - current_iter / max_iter
        lr = self.lr * iter_correction
        sigma = self.sigma * iter_correction

        #Find best matching unit
        loss = self.forward(input)

        # distance_squares = self.locations.float() - bmu_locations.float()
        # distance_squares.pow_(2)
        # distance_squares = torch.sum(distance_squares, dim=2)

        # lr_locations = self._neighborhood_fn(distance_squares, sigma)
        # lr_locations.mul_(lr).unsqueeze_(1)

        delta = lr * (input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0)
        delta.div_(batch_size)
        self.weight.data.add_(delta)

        return loss

    def save_result(self, dir, im_size=(0, 0, 0)):
        '''
        Visualizes the weight of the Self Oranizing Map(SOM)
        :param dir: directory to save
        :param im_size: (channels, size x, size y)
        :return:
        '''
        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1])

        images = images.permute(3, 0, 1, 2)
        save_image(images, dir, normalize=True, padding=1, nrow=self.out_size[0])
=======
	def __init__(self, input_size, out_size=(10, 10), lr=0.3, sigma=None):
		'''
		:param input_size:
		:param out_size:
		:param lr:
		:param sigma:
		'''
		super(SOM, self).__init__()
		self.input_size = input_size
		self.out_size = out_size

		self.lr = lr
		if sigma is None:
			self.sigma = max(out_size) / 2
		else:
			self.sigma = float(sigma)

		self.weight = self.weight_init(inp=input_size,out=out_size[0]*out_size[1],n_samples=5)

		#self.weight = nn.Parameter(torch.Tensor(lhs(input_size, out_size[0] * out_size[1]).T), requires_grad=False)
		self.locations = nn.Parameter(torch.Tensor(list(self.get_map_index())), requires_grad=False)
		self.pdist_fn = nn.PairwiseDistance(p=2)

	def weight_init(self,inp=28*28,out=10*10,n_samples = 5):
		weights = np.zeros((inp,out))
		div_x = int(np.floor(inp/n_samples))
		div_y = int(np.floor(out/n_samples))
		w_val = 1.0/(n_samples*(div_x+div_y))
		
		for i in range(div_x+1):
			for j in range(div_y+1):
				weights[i*n_samples:(i+1)*n_samples,j*n_samples:(j+1)*n_samples] = w_val*(n_samples*i + n_samples*j)
		
		'''
		# Debug
		print("Input Dimension: ", inp)
		print("Output Size: ", out)
		print("Weights Size: ", weights.shape)
		print("N divisions: ",n_samples)
		print("Div x: ", div_x)
		print("Div x: ", div_y)
		print("Initial weight: ", w_val)
		print("Weights: ")
		print(' \n'.join(map(str,weights)))
		'''
		return nn.Parameter(torch.Tensor(weights), requires_grad=False)


	def get_center(i):

	def get_map_index(self):
		'''Two-dimensional mapping function'''
		for i in range(self.out_size[0]):
			for j in range(self.out_size[1]):
				yield (i, j)

	def _neighborhood_fn(self, input, current_sigma):
		'''e^(-(input / sigma^2))'''
		input.div_(current_sigma ** 2)
		input.neg_()
		input.exp_()

		return input

	def forward(self, input):
		'''
		Find the location of best matching unit.
		:param input: data
		:return: location of best matching unit, loss
		'''
		batch_size = input.size()[0]
		input = input.view(batch_size, -1, 1)
		batch_weight = self.weight.expand(batch_size, -1, -1)

		dists = self.pdist_fn(input, batch_weight)
		# Find best matching unit
		losses, bmu_indexes = dists.min(dim=1, keepdim=True)
		bmu_locations = self.locations[bmu_indexes]

		return bmu_locations, bmu_indexes, losses.sum().div_(batch_size).item()

		

	def self_organizing(self, input, current_iter, max_iter):
		'''
		Train the Self Oranizing Map(SOM)
		:param input: training data
		:param current_iter: current epoch of total epoch
		:param max_iter: total epoch
		:return: loss (minimum distance)
		'''
		batch_size = input.size()[0]
		#Set learning rate
		iter_correction = 1.0 - current_iter / max_iter
		lr = self.lr * iter_correction
		sigma = self.sigma * iter_correction

		#Find best matching unit
		bmu_locations, bmu_indexes, loss = self.forward(input)

		distance_squares = self.locations.float() - bmu_locations.float()
		distance_squares.pow_(2)
		distance_squares = torch.sum(distance_squares, dim=2)

		#print("Distance Squares: ", distance_squares)
		#input()

		lr_locations = self._neighborhood_fn(distance_squares, sigma)
		lr_locations.mul_(lr).unsqueeze_(1)


		#print(input.unsqueeze(2).shape, self.weight.shape)

		delta = lr_locations * (input.unsqueeze(2) - self.weight)
		delta = delta.sum(dim=0)
		delta.div_(batch_size)
		self.weight.data.add_(delta)
		return loss, bmu_locations


	def save_result(self, dir, im_size=(0, 0, 0)):
		'''
		Visualizes the weight of the Self Oranizing Map(SOM)
		:param dir: directory to save
		:param im_size: (channels, size x, size y)
		:return:
		'''

		#print(self.weight.shape)


		images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1])

		images = images.permute(3, 0, 1, 2)
		save_image(images, dir, normalize=True, padding=1, nrow=self.out_size[0])
>>>>>>> parent of af0703c... Fix some problems and add main
