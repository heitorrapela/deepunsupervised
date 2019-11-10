# Author: Pedro Braga <phmb4@cin.ufpe.br>.
import os

from models.som import SOM
from dataloaders.datasets import Datasets
import torch.backends.cudnn as cudnn
import random
from torch.utils.data.dataloader import DataLoader
from os.path import join
import argparse
import metrics
from models.cnn_mnist import Net
import torch.optim as optim
import torch
import torch.nn as nn
from utils import utils
from utils.plot import *
import numpy as np


def train_som(root, dataset_path, parameters, device, use_cuda, workers, out_folder,
              n_max=None, evaluate=False):

    dataset = Datasets(dataset=dataset_path, root_folder=root, flatten=True)

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

        epochs = int(parameters[params + 5])

        manual_seed = int(parameters[params + 6])
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        test_loader = DataLoader(dataset.test_data, shuffle=False)

        if use_cuda:
            torch.cuda.manual_seed_all(manual_seed)
            som.cuda()
            cudnn.benchmark = True

        for epoch in range(epochs):
            for batch_idx, (sample, target) in enumerate(train_loader):
                sample, target = sample.to(device), target.to(device)

                som(sample)
                if batch_idx % args.log_interval == 0:
                    print('{0} id {1} [epoch: {2}]'.format(dataset_path,
                                                                         int(params / parameters_count),
                                                                         epoch))

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
        som.write_output(join(out_folder, filename), cluster_result)

        if evaluate:
            print('{0} id {1}\tCE: {4:.6f}'.format(dataset_path,
                                                   int(params / parameters_count),
                                                   epoch,
                                                   100. * batch_idx / len(train_loader),
                                                   metrics.cluster.predict_to_clustering_error(true_labels,
                                                                                               predict_labels)))


def weightedMSELoss(output, target, relevance):
    return torch.sum(relevance * (output - target) ** 2)


def train_full_model(root, dataset_path, device, use_cuda, out_folder, epochs):
    dataset = Datasets(dataset=dataset_path, root_folder=root)
    train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset.test_data, shuffle=False)

    model = Net(device=device)

    manual_seed = 1
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    if use_cuda:
        torch.cuda.manual_seed_all(manual_seed)
        model.cuda()
        cudnn.benchmark = True

    torch.manual_seed(1)
    lr = 0.00001
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss(reduction='sum')

    model.train()
    for epoch in range(epochs):

        # Self-Organize
        for batch_idx, (sample, target) in enumerate(train_loader):
            sample, target = sample.to(device), target.to(device)
            model(sample)

        cluster_result, predict_labels, true_labels = model.cluster(test_loader)
        print("Homogeneity: %0.3f" % metrics.cluster.homogeneity_score(true_labels, predict_labels))
        print("Completeness: %0.3f" % metrics.cluster.completeness_score(true_labels, predict_labels))
        print("V-measure: %0.3f" % metrics.cluster.v_measure_score(true_labels, predict_labels))
        print('{0} \tCE: {1:.3f}'.format(dataset_path,
                                         metrics.cluster.predict_to_clustering_error(true_labels, predict_labels)))

        # Self-Organize and Backpropagate
        avg_loss = 0
        s = 0
        for batch_idx, (sample, target) in enumerate(train_loader):
            #  print("id sample: ", batch_idx, " , target:" ,target)

            #  print("***********************************************************************")
            sample, target = sample.to(device), target.to(device)
            optimizer.zero_grad()

            samples_high_at, weights_unique_nodes_high_at, relevances, _ = model(sample)

            if len(samples_high_at) > 0:  #  if only new nodes were created, the loss is zero, no need to backprobagate it
                weights_unique_nodes_high_at = weights_unique_nodes_high_at.view(-1, model.som_input_size)

                # out = loss(samples_high_at, weights_unique_nodes_high_at)
                # print("msel out:", out)
                out = weightedMSELoss(samples_high_at, weights_unique_nodes_high_at, relevances)
                # print("wmes out:", out)
                out.backward()
                optimizer.step()
            else:
                out = 0.0

            # if batch_idx % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss SOM: {:.6f}'.format(epoch,
            #                                                                        batch_idx * len(sample),
            #                                                                        len(train_loader.dataset),
            #                                                                        100. * batch_idx / len(train_loader),
            #                                                                        out))
            avg_loss += out
            s += len(sample)

        samples = None
        t = None
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            samples_high_at, weights_unique_nodes_high_at, relevances, outputs = model(inputs)

            if samples is None:
                samples = outputs.detach().numpy()
                t = targets.detach().numpy()
            else:
                samples = np.append(samples, outputs.detach().numpy(), axis=0)
                t = np.append(t, targets.detach().numpy(), axis=0)

        centers, relevances, ma = model.som.get_prototypes()
        plot_data(samples, t, centers, relevances*0.1)

        print("Epoch: %d avg_loss: %.6f\n" % (epoch, avg_loss/s))

    plot_hold()
    ## Need to change train loader to test loader...
    model.eval()

    print("Train Finished", flush=True)

    cluster_result, predict_labels, true_labels = model.cluster(test_loader)

    if not os.path.exists(join(args.out_folder, args.dataset.split(".arff")[0])):
        os.makedirs(join(args.out_folder, args.dataset.split(".arff")[0]))

    print("Homogeneity: %0.3f" % metrics.cluster.homogeneity_score(true_labels, predict_labels))
    print("Completeness: %0.3f" % metrics.cluster.completeness_score(true_labels, predict_labels))
    print("V-measure: %0.3f" % metrics.cluster.v_measure_score(true_labels, predict_labels))

    filename = dataset_path.split(".arff")[0] + ".results"
    model.write_output(join(out_folder, filename), cluster_result)

    print('{0} \tCE: {1:.3f}'.format(dataset_path,
                                     metrics.cluster.predict_to_clustering_error(true_labels, predict_labels)))


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
    parser.add_argument('--batch-size', type=int, default=2, help='input batch size')

    parser.add_argument('--epochs', type=int, default=80, help='input total epoch')
    parser.add_argument('--input-paths', default=None, help='Input Paths')
    parser.add_argument('--nmax', type=int, default=None, help='number of nodes')
    parser.add_argument('--params-file', default=None, help='Parameters')

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

    input_paths = utils.read_lines(args.input_paths) if args.input_paths is not None else None
    parameters = utils.read_lines(args.params_file) if args.params_file is not None else None
    n_max = args.nmax

    train_full_model(root, dataset_path, device, use_cuda, out_folder, epochs)

    # if input_paths is None:
    #     train_som(root=root, dataset_path=dataset_path, parameters=parameters, device=device, use_cuda=use_cuda,
    #               workers=args.workers, out_folder=out_folder, n_max=n_max, evaluate=args.eval)
    # else:
    #     for i, train_path in enumerate(input_paths):
    #         train_som(root=root, dataset_path=train_path, parameters=parameters, device=device, use_cuda=use_cuda,
    #                   workers=args.workers, out_folder=out_folder, n_max=n_max, evaluate=args.eval)
