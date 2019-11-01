# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch

from models.som import SOM
from dataloaders.datasets import Datasets

from torch.utils.data.dataloader import DataLoader
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from os.path import join

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


def argument_parser():
    parser = argparse.ArgumentParser(description='Self Organizing Map')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--root', type=str, default='raw-dataloaders/', help='Dataset Root folder')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset Name')
    parser.add_argument('--out-folder', type=str, default='results/', help='Folder to output results')
    parser.add_argument('--batch-size', type=int, default=1, help='input batch size')
    parser.add_argument('--epochs', type=int, default=3, help='input total epoch')
    parser.add_argument('--loginterval', type=int, default=1, help='Log Interval')

    # parser.add_argument('--input-paths', help='Input Paths', required=True)
    # parser.add_argument('--params-file', help='Parameters', required=True)
    #
    # parser.add_argument('--nnodes', type=int, help='number of nodes', required=False, default=200)
    # parser.add_argument('--map-path', help='Map Path')
    # parser.add_argument('--subspace', help='Subspace Clustering', action='store_true', required=False)
    # parser.add_argument('--filter-noise', help='Filter Noise', action='store_true', required=False)
    # parser.add_argument('--keep-map', help='Keep Map', action='store_true', required=False)

    return parser.parse_args()


if __name__ == '__main__':

    # Argument Parser
    args = argument_parser()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists(os.path.dirname(args.out_folder)):
        os.makedirs(os.path.dirname(args.out_folder))

    use_cuda = torch.cuda.is_available() and args.cuda

    if use_cuda:
        torch.cuda.init()

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    ngpu = int(args.ngpu)
    # input_paths = read_lines(args.input_paths)
    # results_folder = args.out_folder
    # parameters = read_lines(args.params_file)

    dataset_path = args.dataset
    root = args.root
    batch_size = args.batch_size
    epochs = args.epochs

    # map_path = args.map_path
    # is_subspace = args.subspace
    # filter_noise = args.filter_noise
    # n_nodes = args.nnodes
    # keep_map = args.kee_map

    dataset = Datasets(dataset=dataset_path, root_folder=root)

    train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset.test_data, shuffle=False)

    # som = SOM(input_size=3, device=device)
    som = SOM(input_dim=dataset.dim_flatten, device=device)
    som = som.to(device)

    for epoch in range(epochs):
        for batch_idx, (sample, target) in enumerate(train_loader):
            sample, target = sample.to(device), target.to(device)

            # sample = torch.tensor([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.], [5., 5., 5.]])
            # target = torch.tensor([1, 2, 3, 4, 5])

            som_loss = som.self_organize(sample)  # Faz forward e ajuste
            if batch_idx % args.loginterval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss SOM: {:.6f}'.format(
                    epoch, batch_idx * len(sample), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), som_loss))

    ## Need to change train loader to test loader...
    cluster_result, predict_labels, true_labels = som.cluster(test_loader)

    if not os.path.exists(join(args.out_folder, args.dataset.split(".arff")[0])):
        os.makedirs(join(args.out_folder, args.dataset.split(".arff")[0]))

    som.write_output(join(args.out_folder,
                          join(args.dataset.split(".arff")[0], args.dataset.split(".arff")[0] + ".results")),
                     cluster_result)

    # args.dataset.split(".arff")[0].split("/")[-1] + ".results", cluster_result)
	# print(np.asarray(predict_labels).shape,np.asarray(true_labels).shape)
	# print(adjusted_rand_score(true_labels,predict_labels))
	# print(completeness_score(true_labels,predict_labels))

    print("Homogeneity: %0.3f" % homogeneity_score(true_labels, predict_labels))
    print("Completeness: %0.3f" % completeness_score(true_labels, predict_labels))
    print("V-measure: %0.3f" % v_measure_score(true_labels, predict_labels))