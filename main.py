from Models.som import SOM
from datasets import Datasets

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.cluster import adjusted_rand_score
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score

import os
import argparse
import numpy as np


def argument_parser():
	# Set args
	parser = argparse.ArgumentParser(description='Self Organizing Map')
	parser.add_argument('--dataset', type=str, default='mnist', help='Dataset Name')
	parser.add_argument('--root', type=str, default='Datasets/', help='Dataset Root folder')
	parser.add_argument('--batch-size', type=int, default=1, help='input batch size')
	parser.add_argument('--epochs', type=int, default=3, help='input total epoch')
	parser.add_argument('--loginterval', type=int, default=1, help='Log Interval')
	return parser.parse_args()


if __name__ == '__main__':
	
	# Argument Parser
	args = argument_parser()
	dataset = args.dataset
	root = args.root
	batch_size = args.batch_size
	epochs = args.epochs 
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	use_cuda = True

	dataset = Datasets(dataset=dataset, root_folder=root)

	train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset.test_data, shuffle=False)

	som = SOM(input_size=dataset.dim_flatten)
	som = som.to(device)

	for epoch in range(epochs):
		for batch_idx, (sample, target) in enumerate(train_loader):
			sample, target = sample.to(device), target.to(device)
			som_loss, _ = som.self_organizing(sample, epoch, epochs)  # Faz forward e ajuste
			if batch_idx % args.loginterval == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss SOM: {:.6f}'.format(
					epoch, batch_idx * len(sample), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), som_loss))

	## Need to change train loader to test loader...
	cluster_result, predict_labels, true_labels = som.cluster(test_loader)
	som.write_output(dataset + ".results", cluster_result)
	#print(np.asarray(predict_labels).shape,np.asarray(true_labels).shape)
	#print(adjusted_rand_score(true_labels,predict_labels))
	#print(completeness_score(true_labels,predict_labels))

	print("Homogeneity: %0.3f" % homogeneity_score(true_labels, predict_labels))
	print("Completeness: %0.3f" % completeness_score(true_labels, predict_labels))
	print("V-measure: %0.3f" % v_measure_score(true_labels, predict_labels))