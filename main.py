# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch

from models.som import SOM
from dataloaders.datasets import Datasets
import torch.backends.cudnn as cudnn
import random

from torch.utils.data.dataloader import DataLoader
from os.path import join


import argparse
import numpy as np
import metrics
from models.cnn_mnist import Net
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from utils import utils


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

        manual_seed = int(parameters[params + 6])
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        test_loader = DataLoader(dataset.test_data, shuffle=False)

        if use_cuda:
            torch.cuda.manual_seed_all(manual_seed)
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
        utils.write_som_output(som, join(out_folder, filename), cluster_result)

        if evaluate:
            print('{0} id {1}\tCE: {4:.6f}'.format(dataset_path,
                                                   int(params / parameters_count),
                                                   epoch,
                                                   100. * batch_idx / len(train_loader),
                                                   metrics.cluster.predict_to_clustering_error(true_labels,
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

    input_paths = utils.read_lines(args.input_paths) if args.input_paths is not None else None
    parameters = utils.read_lines(args.params_file)
    n_max = args.nmax

    if input_paths is None:
        train_som(root=root, dataset_path=dataset_path, parameters=parameters, device=device, use_cuda=use_cuda,
                  workers=args.workers, out_folder=out_folder, n_max=n_max, epochs=epochs, evaluate=args.eval)
    else:
        for i, train_path in enumerate(input_paths):
            train_som(root=root, dataset_path=train_path, parameters=parameters, device=device, use_cuda=use_cuda,
                      workers=args.workers, out_folder=out_folder, n_max=n_max, epochs=epochs, evaluate=args.eval)

    dataset = Datasets(dataset=dataset_path, root_folder=root)
    train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset.test_data, shuffle=False)


    model = Net().to(device)

    torch.manual_seed(1)
    lr = 0.000001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    loss = nn.MSELoss(reduction='sum')
    
    n = 0
    model.train()
    for epoch in range(epochs):
        for batch_idx, (sample, target) in enumerate(train_loader):

            #print("id sample: ", batch_idx, " , target:" ,target)

            #print("***********************************************************************")
            sample, target = sample.to(device), target.to(device)
            optimizer.zero_grad()

            #output_cnn = model(sample)
            #samples_high_at, weights_unique_nodes_high_at, loss_som = som(output_cnn)
            #print(sample.shape)

            samples_high_at, weights_unique_nodes_high_at, loss_som, _ = model(sample)

            #return samples_high_at, nodes_high_at, losses.sum().div_(batch_size)
            #return updatable_samples_hight_at,unique_nodes_high_at, self.relevance, losses.sum().div_(batch_size)
            #output = torch.tensor(np.array(output))
            #print(output.shape)
            #print(output)
            #print(target.shape)
            #print(target)
            #print(sample.shape)
            #print(output_cnn.shape)
            #print(output_som.shape)
            #return samples_high_at, nodes_high_at, self.relevance[bool_high_at], losses.sum().div_(batch_size)
            #return updatable_samples_hight_at, self.weights[unique_nodes_high_at], losses.sum().div_(batch_size)
            #print(len(samples_high_at),len(nodes_high_at),len(relevance_high_at), loss_som)

            #print("------------")
            #print(samples_high_at)
            #print("------------")
            #exit(0)
            #print(weights_unique_nodes_high_at)
            #print(weights_unique_nodes_high_at.shape)


            #n = n + 1
            #exit(0)
            #print(loss_som)
            #print(samples_high_at)
            #if(n == 5):
            #    exit(0)
            
            if samples_high_at is not None:  # if only new nodes were created, the loss is zero, no need to backprobagate it

                weights_unique_nodes_high_at = weights_unique_nodes_high_at.view(-1, 2)

                out = loss(weights_unique_nodes_high_at, samples_high_at)#torch.transpose(som.weight,0,1),output.unsqueeze(1))
                #print("------")
                #print(weights_unique_nodes_high_at.shape, samples_high_at.shape)
                #print(out)
                #print("------")
                out.backward()
                optimizer.step()
            else:
                out = 0.0


            #if(n == 5):
            #    exit(0)
            #n = n+1
            #som.self_organizing(output.view(-1, 28 * 28 * 1), epoch, args.epochs+1)

            #print(" AEW ")
            #exit(0)



            #loss = F.nll_loss(output, target)
            
            '''
            loss.backward()
            
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            '''
            # sample = torch.tensor([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.], [5., 5., 5.]])
            # target = torch.tensor([1, 2, 3, 4, 5])
            #print(output.shape)
            #loss = F.nll_loss(output, target)
            #loss.backward()#som.self_organize(output)  # Faz forward e ajuste
            #optimizer.step()
            if batch_idx % args.loginterval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss SOM: {:.6f}'.format(
                    epoch, batch_idx * len(sample), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), out))
                if out>0.8:
                    print('Sample:', samples_high_at, " Prototype:", weights_unique_nodes_high_at)

    ## Need to change train loader to test loader...
    model.eval()

    print("Train Finish", flush=True)

    cluster_result, predict_labels, true_labels = model.cluster(test_loader,model)

    if not os.path.exists(join(args.out_folder, args.dataset.split(".arff")[0])):
        os.makedirs(join(args.out_folder, args.dataset.split(".arff")[0]))

    model.write_output(join(args.out_folder,
                          join(args.dataset.split(".arff")[0], args.dataset.split(".arff")[0] + ".results")),
                     cluster_result)

    # args.dataset.split(".arff")[0].split("/")[-1] + ".results", cluster_result)
	# print(np.asarray(predict_labels).shape,np.asarray(true_labels).shape)
	# print(adjusted_rand_score(true_labels,predict_labels))
	# print(completeness_score(true_labels,predict_labels))

    #cluster.teste(true_labels,predict_labels)
    print("Homogeneity: %0.3f" % metrics.cluster.homogeneity_score(true_labels, predict_labels))
    print("Completeness: %0.3f" % metrics.cluster.completeness_score(true_labels, predict_labels))
    print("V-measure: %0.3f" % metrics.cluster.v_measure_score(true_labels, predict_labels))