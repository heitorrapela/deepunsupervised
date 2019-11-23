# Author: Heitor Rapela Medeiros <hrm@cin.ufpe.br>.
# ffmpeg -framerate 2 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4
import sys
sys.path.append("..")
import os
from models.som import SOM
from datasets.datasets import Datasets
import random
from torch.utils.data.dataloader import DataLoader
import torch
from utils import utils
from utils.plot import *
from sampling.custom_lhs import *
from argument_parser import argument_parser


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


def som_weights_visualization(root, dataset_path, parameters, grid_rows=10, grid_cols=10, lhs_samples=250):
    dataset = Datasets(dataset=dataset_path, root_folder=root, flatten=True)

    for param_set in parameters.itertuples():

        som = SOM(input_dim=dataset.dim_flatten,
                  n_max=grid_rows * grid_cols,
                  at=param_set.at,
                  ds_beta=param_set.ds_beta,
                  eb=param_set.eb,
                  eps_ds=param_set.eps_ds)

        manual_seed = param_set.seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        train_loader = DataLoader(dataset.train_data, batch_size=1, shuffle=True, num_workers=1)

        fig = plt.figure(figsize=(30, 30), constrained_layout=False)
        gs = fig.add_gridspec(1, 3, hspace=0.1, wspace=0.1)
        gs02 = gs[0, 2].subgridspec(grid_rows, grid_cols, wspace=0.1, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0], xticks=np.array([]), yticks=np.array([]))
        ax2 = fig.add_subplot(gs[0, 1], xticks=np.array([]), yticks=np.array([]))

        fig.suptitle('Self-Organizing Map (SOM) - SOM Clustering', fontsize=16)
        for epoch in range(param_set.epochs):
            print('Experiment {} of {} [epoch: {} of {}]'.format(param_set.Index, lhs_samples, epoch, param_set.epochs))
            for batch_idx, (sample, target) in enumerate(train_loader):

                _, bmu_weights, _ = som(sample)
                _, bmu_indexes = som.get_winners(sample)
                ind_max = bmu_indexes.item()
                weights = som.weights[bmu_indexes]

                if dataset.d_in == 1:
                    ax1.imshow(sample.view(dataset.hw_in, dataset.hw_in), cmap='gray')
                    ax2.imshow(weights.view(dataset.hw_in, dataset.hw_in), cmap='gray')
                    images = [image.reshape(dataset.hw_in, dataset.hw_in) for image in som.weights]
                else:
                    ax1.imshow(sample.view(dataset.d_in, dataset.hw_in, dataset.hw_in).numpy().transpose((1, 2, 0)))
                    ax2.imshow(weights.view(dataset.d_in, dataset.hw_in, dataset.hw_in).numpy().transpose((1, 2, 0)))
                    images = [image.reshape(dataset.hw_in, dataset.hw_in, dataset.d_in) for image in som.weights]

                for x in range(grid_rows):
                    for y in range(grid_cols):
                        if ind_max == (10 * y + x):
                            ax3 = fig.add_subplot(gs02[y, x])
                            if dataset.d_in == 1:
                                ax3.imshow(images[10 * y + x], cmap='gray')
                            else:
                                ax3.imshow(images[10 * y + x].view(dataset.d_in,
                                                                   dataset.hw_in,
                                                                   dataset.hw_in).numpy().transpose((1, 2, 0)))
                            ax3.set_xlabel('{label}'.format(label=ind_max))
                            plt.xticks(np.array([]))
                            plt.yticks(np.array([]))
                ax1.set_xlabel("Sample {} of {}".format(batch_idx, len(train_loader)))
                ax2.set_xlabel("Epoch {} of {}".format(epoch, param_set.epochs))
                ax1.set_title('Input Label: {label}'.format(label=target.item()))
                ax2.set_title('SOM BMU: {label}'.format(label=ind_max))
                plt.pause(0.001)


if __name__ == '__main__':
    args = argument_parser()

    out_folder = args.out_folder if args.out_folder.endswith("/") else args.out_folder + "/"
    if not os.path.exists(os.path.dirname(out_folder)):
        os.makedirs(os.path.dirname(out_folder), exist_ok=True)

    input_paths = utils.read_lines(args.input_paths) if args.input_paths is not None else None

    params_file_som = args.params_file if args.params_file is not None else "../arguments/default_som.lhs"

    if args.lhs:
        parameters = run_lhs_som(params_file_som, args.lhs_samples)
    else:
        parameters = utils.read_params(params_file_som)
    
    som_weights_visualization(root=os.path.join('..',args.root), dataset_path=args.dataset,
                              parameters=parameters, lhs_samples=args.lhs_samples)