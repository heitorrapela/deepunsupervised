#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016
#
# This code uses Self-Organizing Map (SOM) to classify six colours.
# For each epoch it is possible to save an image which represents the weights of the SOM.
# Each weight is a 3D numpy array with values ranging between 0 and 1. The values can be converted
# to RGB in the range [0,255] and then displayed as colours.
# You can use avconv to convert the images to a video: avconv -f image2 -i %d.png -r 12 -s 800x600 output.avi
# The name of the images must be in order, if there is one or more missing names (ex: 18.png, 25.png) 
# an empty video will be created.
# At the end of the example the network is saved inside the file: examples/som_colours.npz

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 

#It requires the pyERA library
#from pyERA.som import Som
#from pyERA.utils import ExponentialDecay
#from pyERA.utils import LinearDecay


import os

from models.som import SOM
from dataloaders.datasets import Datasets
import torch.backends.cudnn as cudnn
import random
from torch.utils.data.dataloader import DataLoader
import argparse
import metrics
from models.cnn_mnist import Net
import torch.optim as optim
import torch
import torch.nn as nn
from utils import utils
from utils.plot import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os.path import join
from sampling.custom_lhs import *
from cuml.manifold import TSNE as cumlTSNE


def train_som(root, dataset_path, parameters, device, use_cuda, workers,
              out_folder, n_max=None, evaluate=False, summ_writer=None):
    dataset = Datasets(dataset=dataset_path, root_folder=root, flatten=True)

    for param_set in parameters.itertuples():
        n_max_som = param_set.n_max if n_max is None else n_max

        som = SOM(input_dim=dataset.dim_flatten,
                  n_max=n_max_som,
                  at=param_set.at,
                  ds_beta=param_set.ds_beta,
                  eb=param_set.eb,
                  eps_ds=param_set.eps_ds,
                  device=device)
        som_epochs = param_set.epochs

        manual_seed = param_set.seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        if use_cuda:
            torch.cuda.manual_seed_all(manual_seed)
            som.cuda()
            cudnn.benchmark = True

        train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        test_loader = DataLoader(dataset.test_data, shuffle=False)

        for epoch in range(som_epochs):
            print('{} [epoch: {}]'.format(dataset_path, epoch))

            for batch_idx, (sample, target) in enumerate(train_loader):
                sample, target = sample.to(device), target.to(device)

                som(sample)
                # if batch_idx % args.log_interval == 0:
                #     print('{0} id {1} [epoch: {2}]'.format(dataset_path, param_set.Index, epoch))

                # if evaluate and batch_idx % args.eval_interval == 0:
                #     _, predict_labels, true_labels = som.cluster(test_loader)
                #     print('{0} id {1} [Epoch: {2} {3:.0f}%]\tCE: {4:.6f}'.format(dataset_path,
                #                                                                  param_set.Index,
                #                                                                  epoch,
                #                                                                  100. * batch_idx / len(train_loader),
                #                                                                  metrics.calculate_ce(true_labels,
                #                                                                               predict_labels)))

        cluster_result, predict_labels, true_labels = som.cluster(test_loader)
        filename = dataset_path.split(".arff")[0] + "_" + str(param_set.Index) + ".results"
        som.write_output(join(out_folder, filename), cluster_result)

        if evaluate:
            ce = metrics.cluster.predict_to_clustering_error(true_labels, predict_labels)
            print('{} \t exp_id {} \tCE: {:.3f}'.format(dataset_path, param_set.Index, ce))

            # summ_writer.add_hparams(hparam_dict=dict(param_set._asdict()),
            #                         metric_dict={'CE_' + dataset_path.split(".arff")[0]: ce})




def argument_parser():
    parser = argparse.ArgumentParser(description='Self Organizing Map')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--workers', type=int,  default=0, help='number of data loading workers')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--log-interval', type=int, default=32, help='Log Interval')
    parser.add_argument('--eval', action='store_true', help='enables evaluation')
    parser.add_argument('--eval-interval', type=int, default=32, help='Evaluation Interval')

    parser.add_argument('--root', type=str, default='raw-datasets/', help='Dataset Root folder')
    parser.add_argument('--tensorboard-root', type=str, default='tensorboard/', help='Tensorboard Root folder')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset Name')
    parser.add_argument('--out-folder', type=str, default='results/', help='Folder to output results')
    parser.add_argument('--batch-size', type=int, default=2, help='input batch size')

    parser.add_argument('--input-paths', default=None, help='Input Paths')
    parser.add_argument('--nmax', type=int, default=None, help='number of nodes')

    parser.add_argument('--lhs', action='store_true', help='enables lhs sampling before run')
    parser.add_argument('--lhs-samples', type=int, default=250, help='Number of Sets to be Sampled using LHS')
    parser.add_argument('--params-file', default=None, help='Parameters')

    parser.add_argument('--som-only', action='store_true', help='Som-Only Mode')
    parser.add_argument('--debug', action='store_true', help='Enables debug mode')
    parser.add_argument('--n-samples', type=int, default=100, help='Dataset Number of Samples')
    parser.add_argument('--lr-cnn', type=float, default=0.00001, help='Learning Rate of CNN Model')

    return parser.parse_args()


if __name__ == '__main__':
    # Argument Parser
    args = argument_parser()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    out_folder = args.out_folder if args.out_folder.endswith("/") else args.out_folder + "/"
    if not os.path.exists(os.path.dirname(out_folder)):
        os.makedirs(os.path.dirname(out_folder), exist_ok=True)

    tensorboard_root = args.tensorboard_root
    if not os.path.exists(os.path.dirname(tensorboard_root)):
        os.makedirs(os.path.dirname(tensorboard_root), exist_ok=True)

    tensorboard_folder = join(tensorboard_root, datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    writer = SummaryWriter(tensorboard_folder)
    print("tensorboard --logdir=" + tensorboard_folder)

    use_cuda = torch.cuda.is_available() and args.cuda

    if use_cuda:
        torch.cuda.init()

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    ngpu = int(args.ngpu)

    root = args.root
    dataset_path = args.dataset
    batch_size = args.batch_size
    debug = args.debug
    n_samples = args.n_samples
    lr_cnn = args.lr_cnn

    input_paths = utils.read_lines(args.input_paths) if args.input_paths is not None else None
    n_max = args.nmax

    if args.som_only:
        params_file_som = args.params_file if args.params_file is not None else "arguments/default_som.lhs"

        if args.lhs:
            parameters = run_lhs_som(params_file_som, args.lhs_samples)
        else:
            parameters = utils.read_params(params_file_som)

        if input_paths is None:
            train_som(root=root, dataset_path=dataset_path, parameters=parameters, device=device,
                      use_cuda=use_cuda, workers=args.workers, out_folder=out_folder, n_max=n_max,
                      evaluate=args.eval, summ_writer=writer)
        else:
            for i, train_path in enumerate(input_paths):
                train_som(root=root, dataset_path=train_path, parameters=parameters, device=device,
                          use_cuda=use_cuda, workers=args.workers, out_folder=out_folder, n_max=n_max,
                          evaluate=args.eval, summ_writer=writer)



def main():


    manual_seed = 1
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    #Set to True if you want to save the SOM images inside a folder.
    SAVE_IMAGE = True
    output_path = "./output/" #Change this path to save in a different forlder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #Init the SOM
    som_size = 512

    my_som = SOM(input_dim=3,
                  n_max=10,
                  device=device)
    #som_epochs = param_set.epochs




    #my_som = Som(matrix_size=som_size, input_size=3, low=0, high=1, round_values=False)
    
    #Init the parameters
    tot_epoch = 1500
    #my_learning_rate = ExponentialDecay(starter_value=0.4, decay_step=50, decay_rate=0.9, staircase=True)
    #my_radius = ExponentialDecay(starter_value=np.rint(som_size/3), decay_step=80, decay_rate=0.90, staircase=True)

    #Starting the Learning
    for epoch in range(1, tot_epoch):

        #Saving the image associated with the SOM weights
        #if(SAVE_IMAGE == True):
        #    img = np.rint(my_som.return_weights_matrix()*255)
        #    plt.axis("off")
        #    plt.imshow(img)
        #    plt.savefig(output_path + str(epoch) + ".png", dpi=None, facecolor='black')

        #Updating the learning rate and the radius
        #learning_rate = my_learning_rate.return_decayed_value(global_step=epoch)
        #radius = my_radius.return_decayed_value(global_step=epoch)

        #Generating random input vectors
        colour_selected = 0#np.random.randint(0, 6)
        colour_range = 100#np.random.randint(100, 255)
        colour_range = float(colour_range) / 255.0
        if(colour_selected == 0): input_vector = np.array([colour_range, 0, 0]) #RED
        if(colour_selected == 1): input_vector = np.array([0, colour_range, 0]) #GREEN
        if(colour_selected == 2): input_vector = np.array([0, 0, colour_range]) #BLUE
        if(colour_selected == 3): input_vector = np.array([colour_range, colour_range, 0]) #YELLOW
        if(colour_selected == 4): input_vector = np.array([0, colour_range, colour_range]) #LIGHT-BLUE
        if(colour_selected == 5): input_vector = np.array([colour_range, 0, colour_range]) #PURPLE


        print(input_vector)
        print(input_vector.shape)
        print(np.transpose(input_vector[:,np.newaxis]).shape)
        #print(input_vector.view(-1, input_vector.shape[0]))
        input_vector = np.transpose(input_vector[:,np.newaxis])

        a, bmu_weights, c= my_som(input_vector)


    #return updatable_samples_hight_at, self.weights[unique_nodes_high_at], self.relevance[unique_nodes_high_at]



    #Saving the network
    file_name = output_path + "som_colours.npz"
    print("Saving the network in: " + str(file_name))
    my_som.save(path=output_path, name="some_colours")

    img = np.rint(my_som.return_weights_matrix()*255)
    plt.axis("off")
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()
