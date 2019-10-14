import torch
import torch.nn as nn
import pandas as pd
import re


class SOM(nn.Module):

    def __init__(self, input_size, out_size=10, lr=0.3, at=0.9, dsbeta=0.0001, eps_ds=0.01, device='cpu'):
        '''
        :param input_size:
        :param out_size:
        :param lr:
        :param at:
        :param dsbeta:
        :param eps_ds:
        :param use_cuda:
        '''

        super(SOM, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.lr = lr
        self.at = at

        self.dsbeta = dsbeta
        self.eps_ds = eps_ds
        self.device = torch.device(device)

        self.node_control = nn.Parameter(torch.zeros(out_size, device=self.device), requires_grad=False)
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

    def pairwise_distance(self, x1, x2=None):

        x1_norm = (x1 ** 2).sum(1).view(-1, 1)

        x2_t = torch.transpose(x2, 0, 1)
        x2_norm = (x2 ** 2).sum(1).view(1, -1)

        dist = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2_t)

        return dist

    def add_node(self, new_samples):
        # number of available nodes in the map
        n_available = torch.tensor(self.node_control[self.node_control == 0].size(0))

        # number of nodes to be inserted in the map
        n_new = torch.tensor(new_samples.size(0))

        # feasible number of nodes to be created
        n_create = torch.min(n_available, n_new)

        # decides the indexes of new samples that will be inserted in fact (less activated to higher activated)
        max_idx = torch.max(torch.tensor(0), torch.tensor(new_samples.size(0)))
        min_idx = torch.max(max_idx - n_create, torch.tensor(0))
        create_idx = torch.arange(start=min_idx, end=max_idx, step=1)
        new_nodes = new_samples[create_idx]

        available_idx = (self.node_control == 0).nonzero().squeeze()

        n_new_nodes = new_nodes.size(0)
        new_nodes_idx = available_idx[:n_new_nodes]

        self.node_control[new_nodes_idx] = 1.
        self.weights[new_nodes_idx] = new_nodes
        self.moving_avg[new_nodes_idx] = nn.Parameter(torch.zeros(n_new_nodes, self.input_size, device=self.device),
                                                      requires_grad=False)
        self.relevance[new_nodes_idx] = nn.Parameter(torch.ones(n_new_nodes, self.input_size, device=self.device),
                                                     requires_grad=False)

        return new_nodes_idx

    def update_node(self, w, index):
        distance = torch.abs(torch.sub(w, self.weights[index]))
        self.moving_avg[index] = torch.mul(self.lr * self.dsbeta, distance) + torch.mul(1 - self.lr * self.dsbeta,
                                                                                   self.moving_avg[index])

        maximum = torch.max(self.moving_avg[index], dim=1, keepdim=True)[0]
        minimum = torch.min(self.moving_avg[index], dim=1, keepdim=True)[0]
        avg = torch.mean(self.moving_avg[index], dim=1, keepdim=True)[0]

        one_tensor = torch.tensor(1, dtype=torch.float, device=self.device)

        self.relevance[index] = torch.div(one_tensor,
                                          one_tensor + torch.exp(torch.div(torch.sub(self.moving_avg[index], avg),
                                                                           torch.mul(self.eps_ds,
                                                                                     torch.sub(maximum, minimum)))))

        delta = torch.mul(self.lr, torch.sub(w, self.weights[index]))
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
        losses = torch.tensor(0)
        nodes_max = torch.tensor([], dtype=torch.long)

        activations = self.activation(input)
        act_max, indexes_max = torch.max(activations, dim=1)

        bool_high_at = act_max >= self.at
        samples_high_at = input[bool_high_at]
        nodes_high_at = indexes_max[bool_high_at]
        if len(nodes_high_at) > 0:
            self.node_control[nodes_high_at] = 1.
            unique_nodes_high_at, updatable_samples_hight_at = self.unique_node_diff_vectorized(nodes_high_at,
                                                                                                samples_high_at)
            losses = self.update_node(updatable_samples_hight_at, nodes_high_at)
            nodes_max = torch.cat((nodes_max, unique_nodes_high_at.view(-1)))

        bool_low_at = act_max < self.at
        samples_low_at = input[bool_low_at]
        nodes_low_at = indexes_max[bool_low_at]
        if len(nodes_low_at) > 0:
            _, updatable_samples_low_at = self.unique_node_diff_vectorized(nodes_low_at, samples_low_at)
            new_nodes_idx = self.add_node(updatable_samples_low_at)
            nodes_max = torch.cat((nodes_max, new_nodes_idx))

        return losses.sum().div_(batch_size), nodes_max

    def unique_node_diff_vectorized(self, nodes, samples):
        unique_nodes, unique_nodes_counts = torch.unique(nodes, return_counts=True)
        unique_nodes = unique_nodes.view(-1, 1)
        repeat_nodes = nodes.repeat(len(unique_nodes), 1)
        unique_nodes_idx = repeat_nodes == unique_nodes
        updatable_samples = torch.matmul(samples.t(), unique_nodes_idx.t().float())
        updatable_samples = torch.div(updatable_samples, unique_nodes_counts.float())

        return unique_nodes, updatable_samples.t()

    def self_organize(self, input):
        '''
        Train the Self Oranizing Map(SOM)
        :param input: training data
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss, location of best matching unit
        '''
        # Find best matching unit
        loss, bmu_indexes = self.forward(input, self.lr)

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
