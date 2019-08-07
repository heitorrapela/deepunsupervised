import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re


class SOM(nn.Module):
    def __init__(self, input_size, out_size=100, lr=0.3, sigma=None, dsbeta=0.0001, eps_ds=0.01,
                 use_cuda=False):
        '''
        :param input_size:
        :param out_size:
        :param lr:
        :param sigma:
        :param dsbeta:
        :param eps_ds:
        :param use_cuda:
        '''

        super(SOM, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.lr = lr

        if sigma is None:
            self.sigma = np.sqrt(out_size) / 2
        else:
            self.sigma = float(sigma)

        self.dsbeta = dsbeta
        self.eps_ds = eps_ds
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.weights = nn.Parameter(torch.zeros(out_size, input_size, device=self.device), requires_grad=False)
        self.moving_avg = nn.Parameter(torch.zeros(out_size, input_size, device=self.device), requires_grad=False)
        self.relevance = nn.Parameter(torch.ones(out_size, input_size, device=self.device), requires_grad=False)

    def activation(self, w):
        dists = self.weighted_distance(w)
        relevance_sum = torch.sum(self.relevance, 1)

        return torch.div(relevance_sum, torch.add(torch.add(relevance_sum, dists), 1e-7))

    def weighted_distance(self, w):
        dists = self.pairwise_distance(w, self.weights)

        relevance_new = self.relevance.sum(1).view(1, -1)

        dist_weight = dists * (relevance_new / self.relevance.size()[-1])

        dist_weight[dist_weight != dist_weight] = 0

        return dist_weight

    def relevance_distance(self, x1, x2):
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)

        return self.pairwise_distance(x1, x2)

    def pairwise_distance(self, x1, x2=None):

        x1_norm = (x1 ** 2).sum(1).view(-1, 1)

        x2_t = torch.transpose(x2, 0, 1)
        x2_norm = (x2 ** 2).sum(1).view(1, -1)

        dist = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2_t)

        dist[dist != dist] = 0  # replace nan values with 0

        return dist

    def update_node(self, w, lr, index):
        distance = torch.abs(torch.sub(w, self.weights[index]))
        self.moving_avg[index] = torch.mul(lr * self.dsbeta, distance) + torch.mul(1 - lr * self.dsbeta,
                                                                                   self.moving_avg[index])

        if len(index.size()) == 0:  # len(index.size()) == 0 means that it is a scalar tensor
            maximum = torch.max(self.moving_avg[index])
            minimum = torch.min(self.moving_avg[index])
            avg = torch.mean(self.moving_avg[index])
        else:
            maximum = torch.max(self.moving_avg[index], 1)[0].unsqueeze(1)
            minimum = torch.min(self.moving_avg[index], 1)[0].unsqueeze(1)
            avg = torch.mean(self.moving_avg[index], 1).unsqueeze(1)

        one_tensor = torch.tensor(1, dtype=torch.float, device=self.device)

        self.relevance[index] = torch.div(one_tensor,
                                          one_tensor + torch.exp(torch.div(torch.sub(self.moving_avg[index], avg),
                                                                           torch.mul(self.eps_ds,
                                                                                     torch.sub(maximum, minimum)))))
        self.relevance[self.relevance != self.relevance] = 1.  # if (max - min) == 0 then set to 1

        delta = torch.mul(lr, torch.sub(w, self.weights[index]))
        self.weights[index] = torch.add(self.weights[index], delta)

        return delta

    def forward(self, input, lr=0.3):
        '''
        Find the location of best matching unit.
        :param input: data
        :param lr: learning rate
        :return: loss, location of best matching unit
        '''
        batch_size = input.size(0)

        losses, indexes_max = self.update_map(input, lr)

        return losses.sum().div_(batch_size), indexes_max

    def update_map(self, w, lr):
        activations = self.activation(w)
        indexes_max = torch.argmax(activations, dim=1)

        return self.update_node(w, lr, indexes_max), indexes_max

    def self_organizing(self, input, current_iter, max_iter):
        '''
        Train the Self Oranizing Map(SOM)
        :param input: training data
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss, location of best matching unit
        '''

        # Set learning rate
        iter_correction = 1.0 - current_iter / max_iter
        lr = self.lr * iter_correction

        # Find best matching unit
        loss, bmu_indexes = self.forward(input, lr)

        return loss, bmu_indexes

    def write_output(self, output_path, result):
        output_file = open(output_path, 'w+')

        content = str(self.weights.size(0)) + "\t" + str(self.weights.size(1)) + "\n"
        self.relevance = nn.Parameter(torch.ones(self.out_size, self.input_size, device=self.device),
                                      requires_grad=False)
        for i, relevance in enumerate(self.relevance.cpu()):
            content += str(i) + "\t" + "\t".join(map(str, relevance.numpy())) + "\n"

        result_text = result.to_string(header=False, index=False)
        result_text = re.sub('\n +', '\n', result_text)
        result_text = re.sub(' +', '\t', result_text)

        content += result_text
        output_file.write(content)
        output_file.close()

    def cluster(self, dataloader):
        clustering = pd.DataFrame(columns=['sample_ind', 'cluster'])
        predict_labels = []
        true_labels = []
        for batch_idx, (inputs, targets) in enumerate(dataloader):

            _, bmu_indexes = self.forward(inputs.to(self.device))
            ind_max = bmu_indexes.item()

            clustering = clustering.append({'sample_ind': batch_idx,
                                            'cluster': ind_max},
                                           ignore_index=True)
            predict_labels.append(ind_max)
            true_labels.append(targets.item())

        return clustering, predict_labels, true_labels
