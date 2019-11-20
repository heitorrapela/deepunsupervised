# Author: Pedro Braga <phmb4@cin.ufpe.br>.

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
# from cuml.manifold import TSNE as cumlTSNE


def train_som(root, dataset_path, parameters, device, use_cuda, workers, out_folder, n_max=None,
              evaluate=False, summ_writer=None, coil20_unprocessed=False):
    dataset = Datasets(dataset=dataset_path, root_folder=root, flatten=True, coil20_unprocessed=coil20_unprocessed)

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


def weightedMSELoss(output, target, relevance):
    return torch.sum(relevance * (output - target) ** 2)


def train_full_model(root, dataset_path, parameters, device, use_cuda,
                     out_folder, debug, n_samples, lr_cnn, summ_writer, print_debug, coil20_unprocessed=False):
    dataset = Datasets(dataset=dataset_path, root_folder=root, debug=debug,
                       n_samples=n_samples, coil20_unprocessed=coil20_unprocessed)

    som_plotter = Plotter()
    tsne_plotter = Plotter()

    # Initialize all meters
    data_timer = utils.Timer()
    batch_timer = utils.Timer()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()

    for param_set in parameters.itertuples():

        model = Net(d_in=dataset.d_in,
                    n_conv_layers=param_set.n_conv,
                    max_pool=True if param_set.max_pool else False,
                    hw_in=dataset.hw_in,
                    som_input=param_set.som_in,
                    filters_list=param_set.filters_pow,
                    kernel_size_list=param_set.n_conv*[param_set.kernel_size],
                    stride_size_list=param_set.n_conv*[1],
                    padding_size_list=param_set.n_conv*[0],
                    max_pool2d_size=param_set.max_pool2d_size,
                    n_max=param_set.n_max,
                    at=param_set.at,
                    eb=param_set.eb,
                    ds_beta=param_set.ds_beta,
                    eps_ds=param_set.eps_ds,
                    device=device)

        manual_seed = param_set.seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        if use_cuda:
            torch.cuda.manual_seed_all(manual_seed)
            model.cuda()
            cudnn.benchmark = True

        train_loader = DataLoader(dataset.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset.test_data, shuffle=False)

        #  optimizer = optim.SGD(model.parameters(), lr=lr_cnn, momentum=0.5)
        optimizer = optim.Adam(model.parameters(), lr=lr_cnn)
        loss = nn.MSELoss(reduction='sum')

        model.train()
        for epoch in range(param_set.epochs):
            # Self-Organize
            # for batch_idx, (sample, target) in enumerate(train_loader):
            #    sample, target = sample.to(device), target.to(device)
            #    model(sample)

            if debug:
                cluster_result, predict_labels, true_labels = model.cluster(test_loader)
                print("Homogeneity: %0.3f" % metrics.cluster.homogeneity_score(true_labels, predict_labels))
                print("Completeness: %0.3f" % metrics.cluster.completeness_score(true_labels, predict_labels))
                print("V-measure: %0.3f" % metrics.cluster.v_measure_score(true_labels, predict_labels))
                nmi = metrics.cluster.nmi(true_labels, predict_labels)
                print("Normalized Mutual Information (NMI): %0.3f" % nmi)
                ari = metrics.cluster.ari(true_labels, predict_labels)
                print("Adjusted Rand Index (ARI): %0.3f" % ari)
                clus_acc = metrics.cluster.acc(true_labels, predict_labels)
                print("Clustering Accuracy (ACC): %0.3f" % clus_acc)
                print('{0} \tCE: {1:.3f}'.format(dataset_path,
                                                 metrics.cluster.predict_to_clustering_error(true_labels,
                                                                                             predict_labels)))

                summ_writer.add_scalar('/NMI', nmi, epoch)
                summ_writer.add_scalar('/ARI', ari, epoch)
                summ_writer.add_scalar('/Acc', clus_acc, epoch)

            # Self-Organize and Backpropagate
            avg_loss = 0
            s = 0
            data_timer.tic()
            batch_timer.tic()
            for batch_idx, (sample, target) in enumerate(train_loader):

                data_time.update(data_timer.toc())  # measure data loading time
                #  print("id sample: ", batch_idx, " , target:" ,target)

                #  print("***********************************************************************")
                sample, target = sample.to(device), target.to(device)
                optimizer.zero_grad()

                samples_high_at, weights_unique_nodes_high_at, relevances = model(sample)

                #  if only new nodes were created, the loss is zero, no need to backprobagate it
                if len(samples_high_at) > 0:
                    weights_unique_nodes_high_at = weights_unique_nodes_high_at.view(-1, model.som_input_size)

                    # out = loss(samples_high_at, weights_unique_nodes_high_at)
                    # print("msel out:", out)
                    out = weightedMSELoss(samples_high_at, weights_unique_nodes_high_at, relevances)
                    # print("wmes out:", out)
                    out.backward()
                    optimizer.step()
                else:
                    out = 0.0

                avg_loss += out
                s += len(sample)
    
                batch_time.update(batch_timer.toc())
                data_timer.toc()

                if print_debug:
                    print('[{0:6d}/{1:6d}]\t'
                    '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    '{data_time.val:.4f} ({data_time.avg:.4f})\t'.format(
                    batch_idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time))

            samples = None
            t = None
            #  Calculate metrics or plot without change SOM map
            if debug:
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model.cnn_extract_features(inputs)

                    if samples is None:
                        samples = outputs.cpu().detach().numpy()
                        t = targets.cpu().detach().numpy()
                    else:
                        samples = np.append(samples, outputs.cpu().detach().numpy(), axis=0)
                        t = np.append(t, targets.cpu().detach().numpy(), axis=0)

                centers, relevances, ma = model.som.get_prototypes()
                som_plotter.plot_data(samples, t, centers.cpu(), relevances.cpu()*0.1)
                summ_writer.add_scalar('Nodes', len(centers), epoch)

                # for center in centers:
                #     t = np.append(t, [10], axis=0)
                # samples = np.append(samples, centers.cpu().detach().numpy(), axis=0)
                # tsne = cumlTSNE(n_components=2, method='barnes_hut')
                # embedding = tsne.fit_transform(samples)
                # tsne_plotter.plot_data(embedding, t, None, None)

            print("Epoch: %d avg_loss: %.6f\n" % (epoch, avg_loss/s))
            summ_writer.add_scalar('Loss/train', avg_loss/s, epoch)

        #  Need to change train loader to test loader...
        model.eval()

        print("Train Finished", flush=True)

        cluster_result, predict_labels, true_labels = model.cluster(test_loader)

        if not os.path.exists(join(args.out_folder, args.dataset.split(".arff")[0])):
            os.makedirs(join(args.out_folder, args.dataset.split(".arff")[0]))

        print("Homogeneity: %0.3f" % metrics.cluster.homogeneity_score(true_labels, predict_labels))
        print("Completeness: %0.3f" % metrics.cluster.completeness_score(true_labels, predict_labels))
        print("V-measure: %0.3f" % metrics.cluster.v_measure_score(true_labels, predict_labels))
        print("Normalized Mutual Information (NMI): %0.3f" % metrics.cluster.nmi(true_labels, predict_labels))
        print("Adjusted Rand Index (ARI): %0.3f" % metrics.cluster.ari(true_labels, predict_labels))
        print("Clustering Accuracy (ACC): %0.3f" % metrics.cluster.acc(true_labels, predict_labels))

        filename = dataset_path.split(".arff")[0] + "_" + str(param_set.Index) + ".results"
        model.write_output(join(out_folder, filename), cluster_result)

        print('{0} \tCE: {1:.3f}'.format(dataset_path,
                                         metrics.cluster.predict_to_clustering_error(true_labels,
                                                                                     predict_labels)))

        if debug:
            som_plotter.plot_hold()
            tsne_plotter.plot_hold()


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
    lhs = FullModelLHS(n_conv=None,
                       som_in=None,
                       max_pool=None,
                       max_pool2d_size=None,
                       filters_pow=None,
                       kernel_size=None,
                       n_max=[10, 200],
                       at=[0.70, 0.999],
                       eb=[0.0001, 0.01],
                       ds_beta=[0.001, 0.5],
                       eps_ds=[0.01, 0.1],
                       epochs=[70, 200],
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
    parser.add_argument('--batch-size', type=int, default=2, help='input batch size')

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

        if input_paths is None:
            train_som(root=root, dataset_path=dataset_path, parameters=parameters, device=device,
                      use_cuda=use_cuda, workers=args.workers, out_folder=out_folder, n_max=n_max,
                      evaluate=args.eval, summ_writer=writer, coil20_unprocessed=coil20_unprocessed)
        else:
            for i, train_path in enumerate(input_paths):
                train_som(root=root, dataset_path=train_path, parameters=parameters, device=device,
                          use_cuda=use_cuda, workers=args.workers, out_folder=out_folder, n_max=n_max,
                          evaluate=args.eval, summ_writer=writer, coil20_unprocessed=coil20_unprocessed)

    else:
        params_file_full = args.params_file if args.params_file is not None else "arguments/default_full_model.lhs"

        if args.lhs:
            parameters = run_lhs_full_model(params_file_full, args.lhs_samples)
        else:
            parameters = utils.read_params(params_file_full)

        train_full_model(root=root, dataset_path=dataset_path, parameters=parameters, device=device, use_cuda=use_cuda,
                         out_folder=out_folder, debug=debug, n_samples=n_samples, lr_cnn=lr_cnn, summ_writer=writer,
                         print_debug=print_debug, coil20_unprocessed=coil20_unprocessed)
