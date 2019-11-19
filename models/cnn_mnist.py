from __future__ import print_function
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.som import SOM


class Net(nn.Module):
    def __init__(self, d_in=1, n_conv_layers=3, max_pool=True, hw_in=28, som_input=2, filters_list=[20, 50],
                 kernel_size_list=[5, 5], stride_size_list=[1, 1], padding_size_list=[0, 0], max_pool2d_size=2,
                 n_max=20, at=0.985, eb=0.1, ds_beta=0.5, eps_ds=1.,  device='cpu'):
        super(Net, self).__init__(),

        self.som_input_size = som_input
        self.d_in = d_in
        self.hw_out = hw_in
        self.max_pool = max_pool
        self.n_conv_layers = n_conv_layers
        self.filters_list = [d_in] + list(np.power(2, self.generate_cnn_filters(filters_list)))
        self.max_pool2d_size = max_pool2d_size
        self.kernel_size_list = kernel_size_list
        self.padding_size_list = padding_size_list
        self.stride_size_list = stride_size_list
        self.convs = []
        for i in range(self.n_conv_layers):
            if not (i < len(self.padding_size_list) and
                    i < len(self.kernel_size_list) and
                    i < len(self.stride_size_list)):
                print("Warning the size of the padding, kernel or stride list is too small!")
                break
            last_hw_out = ((self.hw_out + 2*self.padding_size_list[i]-(self.kernel_size_list[i]-1)-1)//self.stride_size_list[i])+1
            if last_hw_out >= 2:
                self.hw_out = last_hw_out
                self.hw_out = self.hw_out//self.max_pool2d_size if self.max_pool else self.hw_out
                self.convs.append(nn.Sequential(nn.Conv2d(self.filters_list[i],
                                                          self.filters_list[i+1],
                                                          self.kernel_size_list[i],
                                                          self.stride_size_list[i],
                                                          self.padding_size_list[i]),
                                                nn.ReLU(),
                                                nn.MaxPool2d(self.max_pool2d_size,
                                                             self.max_pool2d_size) if self.max_pool else nn.Identity()))
            else:
                print("Warning the size of the output is too small!")
                break
        
        self.fc1 = nn.Linear(self.hw_out*self.hw_out*self.filters_list[len(self.convs)], self.som_input_size)
        self.device = device
        self.som = SOM(input_dim=self.som_input_size, n_max=n_max, eb=eb, at=at, ds_beta=ds_beta, eps_ds=eps_ds, device=self.device)
        self.som = self.som.to(self.device)

    def cnn_extract_features(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
        x = x.view(-1, self.hw_out*self.hw_out*self.filters_list[len(self.convs)])
        x = self.fc1(x)
        x = torch.tanh(x)
        return x

    def forward(self, x):
        return self.som(self.cnn_extract_features(x))
    
    def cluster(self, dataloader):
        clustering = pd.DataFrame(columns=['sample_ind', 'cluster'])
        predict_labels = []
        true_labels = []

        for batch_idx, (samples, targets) in enumerate(dataloader):
            samples, targets = samples.to(self.device), targets.to(self.device)
            outputs = self.cnn_extract_features(samples)

            _, bmu_indexes = self.som.get_winners(outputs.to(self.device))
            ind_max = bmu_indexes.item()
            clustering = clustering.append({'sample_ind': batch_idx,
                                            'cluster': ind_max},
                                           ignore_index=True)
            predict_labels.append(ind_max)
            true_labels.append(targets.item())

            # print("----------------------------------------------")
            # print("Saida CNN: ", outputs)
            # print("Prototipo: ", weights_unique_nodes_high_at)
            # print("Index: ", ind_max)
            # print("----------------------------------------------")

        return clustering, predict_labels, true_labels

    def write_output(self, output_path, cluster_result):
        self.som.write_output(output_path, cluster_result)

    def generate_cnn_filters(self, first_pow, filters_pow_range=[2, 6]):
        power_list = []
        first = 0
        while len(power_list) < self.n_conv_layers:
            if first == 0:
                power_list = power_list + list(range(first_pow, filters_pow_range[-1], 1))
            else:
                power_list = power_list + list(range(filters_pow_range[0], filters_pow_range[-1], 1))
            power_list = power_list + list(range(filters_pow_range[-1], filters_pow_range[0], -1))
            first = 1
        return power_list
