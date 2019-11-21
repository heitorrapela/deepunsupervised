#ffmpeg -framerate 2 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4

import os

from models.som import SOM
from datasets.datasets import Datasets
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
from matplotlib import gridspec


from PIL import Image

def run_lhs_som(filename, lhs_samples=1):
    lhs = SOMLHS(n_max=[5, 10],
                 at=[0.70, 0.999],
                 eb=[0.0001, 0.5],
                 ds_beta=[0.001, 0.8],
                 eps_ds=[0.01, 0.8],
                 epochs=[1, 100],
                 seed=[1, 200000])

    sampling = lhs(lhs_samples)
    lhs.write_params_file(filename)

    return sampling


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
    parser.add_argument('--batch-size', type=int, default=1, help='input batch size')

    parser.add_argument('--input-paths', default=None, help='Input Paths')
    parser.add_argument('--nmax', type=int, default=None, help='number of nodes')

    parser.add_argument('--lhs', action='store_true', help='enables lhs sampling before run')
    parser.add_argument('--lhs-samples', type=int, default=250, help='Number of Sets to be Sampled using LHS')
    parser.add_argument('--params-file', default=None, help='Parameters')

    parser.add_argument('--som-only', action='store_true', help='Som-Only Mode')
    parser.add_argument('--debug', action='store_true', help='Enables debug mode')
    parser.add_argument('--coil20-unprocessed', action='store_true', help='Loads COIL-20 Unprocessed')
    parser.add_argument('--n-samples', type=int, default=100, help='Dataset Number of Samples')
    parser.add_argument('--lr-cnn', type=float, default=0.00001, help='Learning Rate of CNN Model')
    parser.add_argument('--print', action='store_true', help='Print time')

    return parser.parse_args()


def create_color_dataset():
    colors = np.empty((0,3), float)
    colors = np.append(colors, np.array([[0, 0, 0]]), axis=0)
    colors = np.append(colors, np.array([[1, 1, 1]]), axis=0)
    for i in range(10):
        colors = np.append(colors, np.array([[0, 0, random.random()]]), axis=0)
        colors = np.append(colors, np.array([[0, random.random(), 0]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), 0, 0]]), axis=0)
        colors = np.append(colors, np.array([[1, 1, random.random()]]), axis=0)
        colors = np.append(colors, np.array([[1, random.random(), 1]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), 1, 1]]), axis=0)
        colors = np.append(colors, np.array([[0, random.random(), random.random()]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), random.random(), 0]]), axis=0)
        colors = np.append(colors, np.array([[1, random.random(), random.random()]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), random.random(), 1]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), random.random(), random.random()]]), axis=0)
    return torch.Tensor(colors)


def train_som(root, dataset_path, parameters, device, use_cuda, workers, out_folder, n_max=None,
              evaluate=False,grid_rows=10, grid_cols=10, lhs_samples=250):

    dataset = Datasets(dataset=dataset_path, root_folder=root, flatten=True)
    #plt.title('Color SOM')
    #plt.imshow(datat)
    #plt.show()
    #img_to_save = Image.fromarray(weight, "RGB")
    #img_to_save.save('./teste/'+str(param_set.Index) + '.jpg')

    for param_set in parameters.itertuples():


        som = SOM(input_dim=dataset.dim_flatten,
                  n_max=grid_rows*grid_cols,
                  at=0.7,
                  ds_beta=param_set.ds_beta,
                  eb=param_set.eb,
                  eps_ds=param_set.eps_ds)#,
                  #device=device)

        manual_seed = param_set.seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        if use_cuda:
            torch.cuda.manual_seed_all(manual_seed)
            som.cuda()
            cudnn.benchmark = True

        train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        test_loader = DataLoader(dataset.test_data, shuffle=False)


        fig = plt.figure(figsize=(28, 28),constrained_layout=False)
        gs = fig.add_gridspec(1, 3, hspace=0.1, wspace=0.1)
        gs02 = gs[0,2].subgridspec(10,10,wspace=0.1,hspace=0.0)
        ax1 = fig.add_subplot(gs[0,0], xticks=np.array([]),yticks=np.array([]))
        ax2 = fig.add_subplot(gs[0,1], xticks=np.array([]), yticks=np.array([]))

        fig.suptitle('Self-Organizing Map (SOM) - SOM Clustering', fontsize=16)
        epochs = 2#param_set.epochs
        for epoch in range(epochs):
            print('Experiment {} of {} [epoch: {} of {}]'.format(param_set.Index, lhs_samples, epoch,epochs))
            #for idx, inp in enumerate(data):
            for batch_idx, (sample, target) in enumerate(train_loader):
                sample, target = sample.to(device), target.to(device)

                _ , bmu_weights, _ = som(sample)
                _, bmu_indexes = som.get_winners(sample)
                ind_max = bmu_indexes.item()
                weights = som.weights[bmu_indexes]
                
                ax1.imshow(sample.view(28,28), cmap='gray')
                ax2.imshow(weights.view(28,28), cmap='gray')
                images = [image.reshape(28,28) for image in som.weights]
                for x in range(0,10):
                    
                    for y in range(0,10):
                        ax3 = fig.add_subplot(gs02[y,x])
                        ax3.imshow(images[10*(y)+(x)], cmap = 'gray')
                        plt.xticks(np.array([]))
                        plt.yticks(np.array([]))
                
                ax1.set_xlabel("Sample {} of {}".format(batch_idx,len(train_loader)))
                ax2.set_xlabel("Epoch {} of {}".format(epoch,epochs))
                
                ax1.set_title('Input Label: {label}'.format(label=target.item()))
                ax2.set_title('SOM BMU: {label}'.format(label=ind_max))
                ax3.set_xlabel('SOM Nodes (Weights)',x=-4)
                #ax3.set_xlabel("Grid {}x{} of Weights".format(grid_rows,grid_cols))
                plt.pause(0.001)
                
                #plt.savefig('./teste/'+ 'test_' + str(idx) + "_of_" + str(len(data)) + "_epoch_" + str(epoch) + "_of_" + str(param_set.epochs) + '.jpg')
            #plt.show()
        #weights = som.weights
        #weights = weights.reshape(grid_rows,grid_cols,3)
        #plt.title('Color SOM')
        #plt.imshow(weights)
        #plt.savefig('./teste/'+str(param_set.Index) + '.jpg')

        
if __name__ == '__main__':
    # Argument Parser
    args = argument_parser()


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
    coil20_unprocessed = args.coil20_unprocessed
    batch_size = args.batch_size
    debug = args.debug
    n_samples = args.n_samples
    lr_cnn = args.lr_cnn
    print_debug = args.print

    input_paths = utils.read_lines(args.input_paths) if args.input_paths is not None else None
    n_max = args.nmax


    lhs_samples = args.lhs_samples

    input_paths = utils.read_lines(args.input_paths) if args.input_paths is not None else None

    params_file_som = args.params_file if args.params_file is not None else "arguments/default_som.lhs"

    if args.lhs:
        parameters = run_lhs_som(params_file_som, args.lhs_samples)
    else:
        parameters = utils.read_params(params_file_som)


    if input_paths is None:
        train_som(root=root, dataset_path=dataset_path, parameters=parameters, device=device,
                  use_cuda=use_cuda, workers=args.workers, out_folder=out_folder, n_max=n_max,
                  evaluate=args.eval, lhs_samples=lhs_samples)
    else:
        for i, train_path in enumerate(input_paths):
            train_som(root=root, dataset_path=train_path, parameters=parameters, device=device,
                      use_cuda=use_cuda, workers=args.workers, out_folder=out_folder, n_max=n_max,
                      evaluate=args.eval, lhs_samples=lhs_samples)

