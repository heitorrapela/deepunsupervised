# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch

from models.som import SOM
from dataloaders.datasets import Datasets
import torch.backends.cudnn as cudnn
import random

from torch.utils.data.dataloader import DataLoader
from os.path import join

import os
import argparse
import numpy as np
import metrics
import utils

def train_som(root, dataset_path, parameters, device, use_cuda, workers, out_folder,
              n_max=None, epochs=None, evaluate=False):

    dataset = Datasets(dataset=dataset_path, root_folder=root)

    parameters_count = 7

    for params in range(0, len(parameters), parameters_count):

        n_max_som = int(parameters[params]) if n_max is None else n_max

        som = SOM(input_dim=dataset.dim_flatten,
                  n_max=n_max_som,
                  at=float(parameters[params + 1]),
                  dsbeta=float(parameters[params + 2]),
                  lr=float(parameters[params + 3]),
                  eps_ds=float(parameters[params + 4]),
                  device=device)

        epochs_som = int(parameters[params + 5]) if epochs is None else epochs

        manualSeed = int(parameters[params + 6])
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        test_loader = DataLoader(dataset.test_data, shuffle=False)

        if use_cuda:
            torch.cuda.manual_seed_all(manualSeed)
            som.cuda()
            cudnn.benchmark = True

        for epoch in range(epochs_som):
            for batch_idx, (sample, target) in enumerate(train_loader):
                sample, target = sample.to(device), target.to(device)

                som_loss = som.self_organize(sample)
                if batch_idx % args.log_interval == 0:
                    print('{0} id {1} [epoch: {2}] Loss: {3:.6f}'.format(dataset_path,
                                                                         int(params / parameters_count),
                                                                         epoch,
                                                                         som_loss))

                # if evaluate and batch_idx % args.eval_interval == 0:
                #     _, predict_labels, true_labels = som.cluster(test_loader)
                #     print('{0} id {1} [Epoch: {2} {3:.0f}%]\tCE: {4:.6f}'.format(dataset_path,
                #                                                                  int(params / parameters_count),
                #                                                                  epoch,
                #                                                                  100. * batch_idx / len(train_loader),
                #                                                                  metrics.calculate_ce(true_labels,
                #                                                                               predict_labels)))

        cluster_result, predict_labels, true_labels = som.cluster(test_loader)
        filename = dataset_path.split(".arff")[0] + "_" + str(int(params / parameters_count)) + ".results"
        utils.log.write_output(som, join(args.out_folder, filename), cluster_result)

        if evaluate:
            print('{0} id {1}\tCE: {4:.6f}'.format(dataset_path,
                                                   int(params / parameters_count),
                                                   epoch,
                                                   100. * batch_idx / len(train_loader),
                                                   metrics.cluster.calculate_ce(true_labels,
                                                                predict_labels)))


def argument_parser():
    parser = argparse.ArgumentParser(description='Self Organizing Map')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--workers', type=int,  default=0, help='number of data loading workers')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--log-interval', type=int, default=32, help='Log Interval')
    parser.add_argument('--eval', action='store_true', help='enables evaluation')
    parser.add_argument('--eval-interval', type=int, default=32, help='Evaluation Interval')

    parser.add_argument('--root', type=str, default='raw-datasets/', help='Dataset Root folder')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset Name')
    parser.add_argument('--out-folder', type=str, default='results/', help='Folder to output results')
    parser.add_argument('--batch-size', type=int, default=1, help='input batch size')
    parser.add_argument('--epochs', type=int, default=None, help='input total epoch')

    parser.add_argument('--input-paths', default=None, help='Input Paths')
    parser.add_argument('--nmax', type=int, default=None, help='number of nodes')
    parser.add_argument('--params-file', help='Parameters', required=True)

    return parser.parse_args()


if __name__ == '__main__':
    # Argument Parser
    args = argument_parser()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    out_folder = args.out_folder if args.out_folder.endswith("/") else args.out_folder + "/"
    if not os.path.exists(os.path.dirname(out_folder)):
        os.makedirs(os.path.dirname(out_folder), exist_ok=True)

    use_cuda = torch.cuda.is_available() and args.cuda

    if use_cuda:
        torch.cuda.init()

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    ngpu = int(args.ngpu)

    root = args.root
    dataset_path = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs

    input_paths = utils.utils.read_lines(args.input_paths) if args.input_paths is not None else None
    parameters = utils.utils.read_lines(args.params_file)
    n_max = args.nmax

    if input_paths is None:
        train_som(root=root, dataset_path=dataset_path, parameters=parameters, device=device, use_cuda=use_cuda,
                  workers=args.workers, out_folder=out_folder, n_max=n_max, epochs=epochs, evaluate=args.eval)
    else:
        for i, train_path in enumerate(input_paths):
            train_som(root=root, dataset_path=train_path, parameters=parameters, device=device, use_cuda=use_cuda,
                      workers=args.workers, out_folder=out_folder, n_max=n_max, epochs=epochs, evaluate=args.eval)
