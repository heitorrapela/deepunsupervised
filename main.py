# Author: Pedro Braga <phmb4@cin.ufpe.br>.

from argument_parser import argument_parser
from train_som import train_som
from train_full_model import train_full_model
import os

import torch
from utils import utils, results_eval, plot
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os.path import join
from sampling.custom_lhs import *


def run_lhs_som(filename, lhs_samples=1):
    lhs = SOMLHS(n_max=[5, 10],
                 at=[0.70, 0.999],
                 eb=[0.0001, 0.01],
                 ds_beta=[0.001, 0.5],
                 eps_ds=[0.01, 0.1],
                 epochs=[1, 3],
                 seed=[1, 200000])

    sampling = lhs(lhs_samples)
    lhs.write_params_file(filename)

    return sampling


def run_lhs_full_model(filename, lhs_samples=1):
    lhs = FullModelLHS(n_conv=[2, 5],
                       lr_cnn=[0.00001, 0.001],
                       som_in=[10, 100],
                       max_pool=[0, 1],
                       max_pool2d_size=[3, 4],
                       filters_pow=[2, 6],
                       kernel_size=[0.5, 3.5],
                       n_max=[10, 150],
                       at=[0.85, 0.999],
                       eb=[0.0001, 0.005],
                       ds_beta=[0.001, 0.5],
                       eps_ds=[0.01, 0.1],
                       epochs=[70, 200],
                       seed=[1, 200000])

    sampling = lhs(lhs_samples)
    lhs.write_params_file(filename)

    return sampling


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
    coil20_unprocessed = args.coil20_unprocessed
    batch_size = args.batch_size
    debug = args.debug
    n_samples = args.n_samples
    lr_cnn = args.lr_cnn
    print_debug = args.print

    input_paths = utils.read_lines(args.input_paths) if args.input_paths is not None else None
    n_max = args.nmax

    if args.som_only:
        params_file_som = args.params_file if args.params_file is not None else "arguments/default_som.lhs"

        if args.lhs:
            parameters = run_lhs_som(params_file_som, args.lhs_samples)
        else:
            parameters = utils.read_params(params_file_som)

        # plot.HParams().plot_params_results("results/mnist_3000.csv", save=False, plot=True)

        # results_eval.true_to_clustering_error(out_folder, root, "results/uci_test",
        #                                       repeats=4, param_file=params_file_som)

        # results_eval.dataset_to_clustering_error(out_folder, "mnist", root, "results/mnist_3000",
        #                                          debug, n_samples, params_file_som, coil20_unprocessed)

        if input_paths is None:
            train_som(root=root, dataset_path=dataset_path, parameters=parameters, device=device, use_cuda=use_cuda,
                      workers=args.workers, out_folder=out_folder, batch_size=batch_size, n_max=n_max,
                      evaluate=args.eval, summ_writer=writer, coil20_unprocessed=coil20_unprocessed)
        else:
            for i, train_path in enumerate(input_paths):
                train_som(root=root, dataset_path=train_path, parameters=parameters, device=device, use_cuda=use_cuda,
                          workers=args.workers, out_folder=out_folder, batch_size=batch_size, n_max=n_max,
                          evaluate=args.eval, summ_writer=writer, coil20_unprocessed=coil20_unprocessed)


    else:
        params_file_full = args.params_file if args.params_file is not None else "arguments/default_full_model.lhs"

        if args.lhs:
            parameters = run_lhs_full_model(params_file_full, args.lhs_samples)
        else:
            parameters = utils.read_params(params_file_full)

        train_full_model(root=root, dataset_path=dataset_path, parameters=parameters, device=device, use_cuda=use_cuda,
                         out_folder=out_folder, debug=debug, n_samples=n_samples, lr_cnn=lr_cnn, batch_size=batch_size,
                         summ_writer=writer, print_debug=print_debug, coil20_unprocessed=coil20_unprocessed)
