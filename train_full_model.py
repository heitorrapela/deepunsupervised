# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import os
from datasets.datasets import Datasets
import torch.backends.cudnn as cudnn
import random
from torch.utils.data.dataloader import DataLoader
import metrics
from models.cnn_mnist import Net
import torch.optim as optim
import torch
import torch.nn as nn
from utils import utils
from utils.plot import *
from os.path import join
from sampling.custom_lhs import *
# from cuml.manifold import TSNE as cumlTSNE


def weightedMSELoss(output, target, relevance):
    return torch.sum(relevance * (output - target) ** 2)


def train_full_model(root, dataset_path, parameters, device, use_cuda, out_folder, debug, n_samples,
                     lr_cnn, batch_size, summ_writer, print_debug, coil20_unprocessed=False):
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
                    kernel_size_list=param_set.n_conv * [param_set.kernel_size],
                    stride_size_list=param_set.n_conv * [1],
                    padding_size_list=param_set.n_conv * [0],
                    max_pool2d_size=param_set.max_pool2d_size,
                    n_max=param_set.n_max,
                    at=param_set.at,
                    eb=param_set.eb,
                    ds_beta=param_set.ds_beta,
                    eps_ds=param_set.eps_ds,
                    ld=param_set.ld,
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
        optimizer = optim.Adam(model.parameters(), lr=param_set.lr_cnn)
        loss = nn.MSELoss(reduction='sum')

        model.train()
        for epoch in range(param_set.epochs):

            # Self-Organize and Backpropagate
            avg_loss = 0
            s = 0
            data_timer.tic()
            batch_timer.tic()
            for batch_idx, (sample, target) in enumerate(train_loader):

                data_time.update(data_timer.toc())
                sample, target = sample.to(device), target.to(device)
                optimizer.zero_grad()
                samples_high_at, weights_unique_nodes_high_at, relevances = model(sample)

                #  if only new nodes were created, the loss is zero, no need to backprobagate it
                if len(samples_high_at) > 0:
                    weights_unique_nodes_high_at = weights_unique_nodes_high_at.view(-1, model.som_input_size)
                    out = weightedMSELoss(samples_high_at, weights_unique_nodes_high_at, relevances)
                    out.backward()
                    optimizer.step()
                else:
                    out = 0.0

                avg_loss += out
                s += len(sample)

                batch_time.update(batch_timer.toc())
                data_timer.toc()

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

                    if summ_writer is not None:
                        summ_writer.add_scalar('/NMI', nmi, epoch)
                        summ_writer.add_scalar('/ARI', ari, epoch)
                        summ_writer.add_scalar('/Acc', clus_acc, epoch)

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
                som_plotter.plot_data(samples, t, centers.cpu(), relevances.cpu() * 0.1)

                if summ_writer is not None:
                    summ_writer.add_scalar('Nodes', len(centers), epoch)

                # for center in centers:
                #     t = np.append(t, [10], axis=0)
                # samples = np.append(samples, centers.cpu().detach().numpy(), axis=0)
                # tsne = cumlTSNE(n_components=2, method='barnes_hut')
                # embedding = tsne.fit_transform(samples)
                # tsne_plotter.plot_data(embedding, t, None, None)

            print("Epoch: %d avg_loss: %.6f\n" % (epoch, avg_loss / s))
            if summ_writer is not None:
                summ_writer.add_scalar('Loss/train', avg_loss / s, epoch)

        #  Need to change train loader to test loader...
        model.eval()

        print("Train Finished", flush=True)

        cluster_result, predict_labels, true_labels = model.cluster(test_loader)

        if not os.path.exists(join(out_folder, dataset_path.split(".arff")[0])):
            os.makedirs(join(out_folder, dataset_path.split(".arff")[0]))

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
            # tsne_plotter.plot_hold()

