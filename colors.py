#ffmpeg -framerate 2 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from models.som import SOM
from sampling.custom_lhs import *
import argparse
import os
from os.path import join
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from utils.plot import *
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
    parser.add_argument('--out-folder', type=str, default='results/', help='Folder to output results')
    parser.add_argument('--input-paths', default=None, help='Input Paths')
    parser.add_argument('--lhs', action='store_true', help='enables lhs sampling before run')
    parser.add_argument('--lhs-samples', type=int, default=250, help='Number of Sets to be Sampled using LHS')
    parser.add_argument('--params-file', default=None, help='Parameters')
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


def train_som(parameters, out_folder, grid_rows=10,grid_cols=10, lhs_samples=250):

    # Set color dataset
    data = create_color_dataset()
    
    #plt.title('Color SOM')
    #plt.imshow(datat)
    #plt.show()
    #img_to_save = Image.fromarray(weight, "RGB")
    #img_to_save.save('./teste/'+str(param_set.Index) + '.jpg')

    for param_set in parameters.itertuples():

        som = SOM(input_dim=3,
                  n_max=grid_rows*grid_cols,
                  at=param_set.at,
                  ds_beta=param_set.ds_beta,
                  eb=param_set.eb,
                  eps_ds=param_set.eps_ds)

        manual_seed = param_set.seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        fig = plt.figure(figsize=(grid_rows, 3*grid_cols))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.set_title('Input Color')
        ax2.set_title('SOM BMU')
        ax3.set_title('SOM Nodes (Weights)')
        
        fig.suptitle('Self-Organizing Map (SOM) - Color Clustering', fontsize=16)
        epochs = 2#param_set.epochs
        for epoch in range(epochs):
            print('Experiment {} of {} [epoch: {} of {}]'.format(param_set.Index, lhs_samples, epoch,epochs))
            for idx, inp in enumerate(data):
                #print(inp)
                inp = inp.view(-1,3)
                _ , bmu_weights, _ = som(inp)
                _, bmu_indexes = som.get_winners(inp)
                ind_max = bmu_indexes.item()
                weights = som.weights[bmu_indexes]
                
                ax1.imshow(inp.view(-1,1,3))
                ax2.imshow(weights.view(-1,1,3))
                ax3.imshow(som.weights.view(grid_rows,grid_cols,3))
                ax1.set_xlabel("Sample {} of {}".format(idx,len(data)))
                ax2.set_xlabel("Epoch {} of {}".format(epoch,epochs))
                ax3.set_xlabel("Grid {}x{} of Weights".format(grid_rows,grid_cols))


                plt.pause(0.001)
                #plt.savefig('./teste/'+ 'test_' + str(idx) + "_of_" + str(len(data)) + "_epoch_" + str(epoch) + "_of_" + str(param_set.epochs) + '.jpg')

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


    lhs_samples = args.lhs_samples

    input_paths = utils.read_lines(args.input_paths) if args.input_paths is not None else None

    params_file_som = args.params_file if args.params_file is not None else "arguments/default_som.lhs"

    if args.lhs:
        parameters = run_lhs_som(params_file_som, args.lhs_samples)
    else:
        parameters = utils.read_params(params_file_som)

    if input_paths is None:
        train_som(parameters=parameters, out_folder=out_folder,
                  lhs_samples=lhs_samples)
    else:
        for i, train_path in enumerate(input_paths):
            train_som(parameters=parameters, out_folder=out_folder,
                      lhs_samples=lhs_samples)
