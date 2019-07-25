from Models.som import SOM
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

if __name__ == '__main__':
	batch_size = 10
	epochs = 10

	transform = transforms.Compose(
		[transforms.ToTensor()]
	)
	
	train_data = datasets.MNIST(root='Datasets/',train=True, download=True, transform=transform)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	som_input_dim = 28*28
	som = SOM(input_size=som_input_dim, out_size=(28,28))
	som = som.to(device)

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		som_loss = som.forward(data)
		som.self_organizing(data.view(-1, 28 * 28 * 1), 0, epochs)