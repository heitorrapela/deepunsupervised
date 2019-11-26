# Author: Pedro Braga <phmb4@cin.ufpe.br>.

from models.som import SOM
from datasets.datasets import Datasets
import torch.backends.cudnn as cudnn
import random
from torch.utils.data.dataloader import DataLoader
import metrics
import torch
from utils.plot import *
from os.path import join


def train_som(root, dataset_path, parameters, device, use_cuda, workers, out_folder, batch_size, n_max=None,
              evaluate=False, summ_writer=None, coil20_unprocessed=False):
    dataset = Datasets(dataset=dataset_path, root_folder=root, flatten=True, coil20_unprocessed=coil20_unprocessed)

    plots = HParams()
    clustering_errors = []
    for param_set in parameters.itertuples():
        n_max_som = param_set.n_max if n_max is None else n_max

        som = SOM(input_dim=dataset.dim_flatten,
                  n_max=n_max_som,
                  at=param_set.at,
                  ds_beta=param_set.ds_beta,
                  eb=param_set.eb,
                  eps_ds=param_set.eps_ds,
                  ld=param_set.ld,
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

        cluster_result, predict_labels, true_labels = som.cluster(test_loader)
        filename = dataset_path.split(".arff")[0] + "_" + str(param_set.Index) + ".results"
        som.write_output(join(out_folder, filename), cluster_result)

        if evaluate:
            ce = metrics.cluster.predict_to_clustering_error(true_labels, predict_labels)
            clustering_errors.append(ce)
            print('{} \t exp_id {} \tCE: {:.3f}'.format(dataset_path, param_set.Index, ce))

    if evaluate and summ_writer is not None:
        clustering_errors = np.array(clustering_errors)
        plots.plot_tensorboard_x_y(parameters, 'CE', clustering_errors, summ_writer, dataset_path.split(".arff")[0])

