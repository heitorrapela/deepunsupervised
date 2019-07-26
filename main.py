from Models.som import SOM
from datasets import ArffDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

import os
import argparse
import numpy as np

def argumentParser():
	# Set args
	parser = argparse.ArgumentParser(description='Self Organizing Map')
	parser.add_argument('--dataset', type=str, default='mnist', help='Dataset Name')
	parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
	parser.add_argument('--epochs', type=int, default=3, help='input total epoch')
	parser.add_argument('--row', type=int, default=10, help='set SOM row length')
	parser.add_argument('--col', type=int, default=10, help='set SOM col length')
	parser.add_argument('--loginterval', type=int, default=1, help='Log Interval')
	return parser.parse_args()


if __name__ == '__main__':
	
	# Argument Parser
	args = argumentParser()	
	dataset = args.dataset
	batch_size = args.batch_size
	epochs = args.epochs 
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Varia de acordo com o dataset ou features, deixei assim por enquanto :)
	som_input_height = 28
	som_input_width = 28
	som_input_deep = 1
	som_input_dim = som_input_height*som_input_width*som_input_deep

	transform = transforms.Compose(
		[transforms.ToTensor()]
	)
	
	train_data = None
	train_loader = None

	if(dataset == "mnist"):
		train_data = datasets.MNIST(root='Datasets/',train=True, download=True, transform=transform)
		train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	elif(dataset == "fashion"):
		train_data = datasets.FashionMNIST(root='Datasets/',train=True, download=True, transform=transform)
		train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	elif(dataset == "custom"):
		test = 'a'

	som = SOM(input_size=som_input_dim, out_size=(args.row,args.col))
	som = som.to(device)

	for epoch in range(epochs):
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)
			som_loss = som.forward(data) # Se quiser adicionar a loss com outra
			som.self_organizing(data.view(-1, som_input_height * som_input_width * som_input_deep), epoch, epochs) # Faz forward e ajuste

			if batch_idx % args.loginterval == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss SOM: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), som_loss))

