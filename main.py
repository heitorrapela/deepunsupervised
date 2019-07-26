from Models.som import SOM
from datasets import ArffDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score

import os
import argparse
import numpy as np

def read_lines(file_path):
    if os.path.isfile(file_path):
        data = open(file_path, 'r')
        data = np.array(data.read().splitlines())
    else:
        data = []

    return data

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
	use_cuda = True

	transform = transforms.Compose(
		[transforms.ToTensor()]
	)
	
	train_data = None
	train_loader = None
	test_data = None
	test_loader = None

	if(dataset == "mnist"):
		train_data = datasets.MNIST(root='Datasets/',train=True, download=True, transform=transform)
		train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
		test_data = datasets.MNIST(root='Datasets/',train=False, download=True, transform=transform)
		test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
		som_input_height = 28
		som_input_width = 28
		som_input_deep = 1

	elif(dataset == "fashion"):
		train_data = datasets.FashionMNIST(root='Datasets/',train=True, download=True, transform=transform)
		train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
		test_data = datasets.FashionMNIST(root='Datasets/',train=False, download=True, transform=transform)
		test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
		som_input_height = 28
		som_input_width = 28
		som_input_deep = 1
	elif(dataset == "custom"):
		## Custom data
		train = 'Datasets/Realdata/breast.arff'
		train_data = ArffDataset(train)
		train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

		test_data = train_data
		test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

		som_input_height = 33
		som_input_width = 1
		som_input_deep = 1

	# Varia de acordo com o dataset ou features, deixei assim por enquanto :)
	som_input_dim = som_input_height*som_input_width*som_input_deep

	som = SOM(input_size=som_input_dim, out_size=(args.row,args.col),use_cuda=use_cuda)
	som = som.to(device)

	for epoch in range(epochs):
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)
			som_loss, _ = som.forward(data) # Se quiser adicionar a loss com outra
			som.self_organizing(data.view(-1, som_input_dim), epoch, epochs) # Faz forward e ajuste
			if batch_idx % args.loginterval == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss SOM: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), som_loss))

	## Need to change train loader to test loader...
	cluster_result, predict_labels,true_labels = som.cluster(test_loader)
	som.write_output(dataset +  ".results",cluster_result)
	#print(np.asarray(predict_labels).shape,np.asarray(true_labels).shape)
	#print(adjusted_rand_score(true_labels,predict_labels))
	#print(completeness_score(true_labels,predict_labels))

	print("Homogeneity: %0.3f" % homogeneity_score(true_labels,predict_labels))
	print("Completeness: %0.3f" % completeness_score(true_labels,predict_labels))
	print("V-measure: %0.3f" % v_measure_score(true_labels,predict_labels))