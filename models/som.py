# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch
import torch.nn as nn
import pandas as pd
import re


class SOM(nn.Module):

    def __init__(self, input_dim, n_max=20, eb=0.1, at=0.985, ds_beta=0.5, eps_ds=1., ld=0.1, device='cpu'):
        '''
        :param input_dim:
        :param n_max:
        :param at:
        :param ds_beta:
        :param eb:
        :param eps_ds:
        :param ld:
        :param use_cuda:
        '''

        super(SOM, self).__init__()
        self.input_size = input_dim

        self.n_max = n_max
        self.lr = eb
        self.at = at
        self.ds_beta = ds_beta
        self.eps_ds = eps_ds
        self.ld = ld

        self.device = torch.device(device)

        self.node_control = nn.Parameter(torch.zeros(n_max, device=self.device), requires_grad=False)
        self.weights = nn.Parameter(torch.zeros(n_max, input_dim, device=self.device), requires_grad=False)
        self.moving_avg = nn.Parameter(torch.zeros(n_max, input_dim, device=self.device), requires_grad=False)
        self.relevance = nn.Parameter(torch.ones(n_max, input_dim, device=self.device), requires_grad=False)
        self.life = nn.Parameter(torch.ones(n_max, device=self.device), requires_grad=False)

    def activation(self, w):
        dists = self.weighted_distance(w)
        relevance_sum = torch.sum(self.relevance, 1)

        return torch.div(relevance_sum, torch.add(torch.add(relevance_sum, dists), 1e-3))

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
        available_idx = (self.node_control == 0).nonzero()
        n_available = torch.tensor(available_idx.size(0))

        # number of nodes to be inserted in the map
        n_new = torch.tensor(new_samples.size(0))

        # feasible number of nodes to be created
        n_create = torch.min(n_available, n_new)

        # decides the indexes of new samples that will be inserted in fact (less activated to higher activated)
        max_idx = torch.max(torch.tensor(0), torch.tensor(new_samples.size(0)))
        min_idx = torch.max(max_idx - n_create, torch.tensor(0))
        create_idx = torch.arange(start=min_idx, end=max_idx, step=1)
        new_nodes = new_samples[create_idx]

        n_new_nodes = new_nodes.size(0)
        new_nodes_idx = available_idx[:n_new_nodes].t()

        self.node_control[new_nodes_idx] = 1.
        self.life[new_nodes_idx] = 1.
        self.weights[new_nodes_idx] = new_nodes
        self.relevance[new_nodes_idx] = nn.Parameter(torch.ones(n_new_nodes, self.input_size, device=self.device),
                                                     requires_grad=False)

        self.moving_avg[new_nodes_idx] = nn.Parameter(torch.zeros(n_new_nodes, self.input_size, device=self.device),
                                                      requires_grad=False)

        return new_nodes_idx.squeeze(-1)

    def update_node(self, w, index):
        distance = torch.abs(torch.sub(w, self.weights[index]))
        self.moving_avg[index] = torch.mul(self.lr * self.ds_beta, distance) + torch.mul(1 - (self.lr * self.ds_beta),
                                                                                         self.moving_avg[index])

        maximum = torch.max(self.moving_avg[index], dim=1, keepdim=True)[0]
        minimum = torch.min(self.moving_avg[index], dim=1, keepdim=True)[0]
        avg = torch.mean(self.moving_avg[index], dim=1, keepdim=True)

        one_tensor = torch.tensor(1, dtype=torch.float, device=self.device)

        self.relevance[index] = torch.div(one_tensor,
                                          one_tensor + torch.exp(torch.div(torch.sub(self.moving_avg[index], avg),
                                                                           torch.mul(self.eps_ds,
                                                                                     torch.sub(maximum, minimum)))))
        # print("relevances:", self.relevance)
        # if (max - min) == 0 or (mv_avg - avg) == 0 then set to 1
        self.relevance[self.relevance != self.relevance] = 1.

        delta = torch.mul(self.lr, torch.sub(w, self.weights[index]))
        self.weights[index] = torch.add(self.weights[index], delta)

    def get_winners(self, input):
        activations = self.activation(input) * self.node_control
        return torch.max(activations, dim=1)

    def get_prototypes(self):
        mask = self.node_control != 0
        return self.weights[mask], self.relevance[mask], self.moving_avg[mask]

    def forward(self, input, lr=0.01):
        '''
        Find the location of best matching unit.
        :param input: data
        :param lr: learning rate
        :return: loss, location of best matching unit
        '''
        act_max, indexes_max = self.get_winners(input)

        self.life -= self.ld

        bool_high_at = act_max >= self.at
        samples_high_at = input[bool_high_at]
        nodes_high_at = indexes_max[bool_high_at]

        updatable_samples_hight_at = []
        unique_nodes_high_at = []
        if len(nodes_high_at) > 0:
            self.node_control[nodes_high_at] = 1.
            unique_nodes_high_at, updatable_samples_hight_at = self.unique_node_diff_vectorized(nodes_high_at,
                                                                                                samples_high_at)

            self.life[unique_nodes_high_at] = 1.
            with torch.no_grad():
                self.update_node(updatable_samples_hight_at, unique_nodes_high_at)

        bool_low_at = act_max < self.at
        samples_low_at = input[bool_low_at]
        nodes_low_at = indexes_max[bool_low_at]
        
        # if there is nodes to be inserted and positions available in the map
        if len(nodes_low_at) > 0 and self.node_control[self.node_control == 0].size(0) > 0:
            _, updatable_samples_low_at = self.unique_node_diff_vectorized(nodes_low_at, samples_low_at)

            with torch.no_grad():
                self.add_node(updatable_samples_low_at)

        self.remove_nodes()

        return updatable_samples_hight_at, self.weights[unique_nodes_high_at], self.relevance[unique_nodes_high_at]

    def unique_node_diff_vectorized(self, nodes, samples):
        unique_nodes, unique_nodes_counts = torch.unique(nodes, return_counts=True)
        unique_nodes = unique_nodes.view(-1, 1)
        repeat_nodes = nodes.repeat(len(unique_nodes), 1)
        unique_nodes_idx = repeat_nodes == unique_nodes
        updatable_samples = torch.matmul(samples.t(), unique_nodes_idx.t().float())
        updatable_samples = torch.div(updatable_samples, unique_nodes_counts.float())

        return unique_nodes.t().squeeze(-1), updatable_samples.t()

    def remove_nodes(self):
        dead_nodes = self.life <= 0.
        self.node_control[dead_nodes] = 0.

    def cluster(self, dataloader):
        clustering = pd.DataFrame(columns=['sample_ind', 'cluster'])
        predict_labels = []
        true_labels = []

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            _, bmu_indexes = self.get_winners(inputs.to(self.device))

            for index, bmu_index in enumerate(bmu_indexes):
                ind_max = bmu_index.item()

                clustering = clustering.append({'sample_ind': batch_idx,
                                                'cluster': ind_max},
                                               ignore_index=True)
                predict_labels.append(ind_max)
                true_labels.append(targets[index].item())

        return clustering, predict_labels, true_labels

    def write_output(self, output_path, cluster_result):
        print(output_path)
        output_file = open(output_path, 'w+')

        n_clusters = self.node_control[self.node_control == 1].size(0)

        content = str(n_clusters) + "\t" + str(self.input_size) + "\n"
        for i, relevance in enumerate(self.relevance):
            if self.node_control[i] == 1:
                with torch.no_grad():
                    content += str(i) + "\t" + "\t".join(map(str, relevance.cpu().numpy())) + "\n"

        result_text = cluster_result.to_string(header=False, index=False).strip()
        result_text = re.sub('\n +', '\n', result_text)
        result_text = re.sub(' +', '\t', result_text)

        content += result_text
        output_file.write(content)
        output_file.close()
