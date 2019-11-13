from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from models.som import SOM


class Net(nn.Module):
    def __init__(self, d_in=1, hw_in=28, som_input=2, filters_list=[20, 50], kernel_size_list=[5, 5],
                 stride_size_list=[1, 1], padding_size_list=[0, 0], max_pool2d_size=2, device='cpu'):
        super(Net, self).__init__(),

        self.som_input_size = som_input
        self.d_in = d_in
        self.hw_in = hw_in
        self.filters_list = filters_list
        self.max_pool2d_size = max_pool2d_size
        self.kernel_size_list = kernel_size_list
        self.padding_size_list = padding_size_list
        self.stride_size_list = stride_size_list

        self.conv1 = nn.Conv2d( self.d_in,
                                self.filters_list[0],
                                self.kernel_size_list[0],
                                self.stride_size_list[0],
                                self.padding_size_list[0])


        self.hw_out = ((self.hw_in + 2*self.padding_size_list[0]-(self.kernel_size_list[0]-1)-1)//self.stride_size_list[0])+1
        self.hw_out = self.hw_out//self.max_pool2d_size

        self.conv2 = nn.Conv2d( self.filters_list[0],
                                self.filters_list[1],
                                self.kernel_size_list[1],
                                self.stride_size_list[1],
                                self.padding_size_list[1])

        self.hw_out = ((self.hw_out + 2*self.padding_size_list[1]-(self.kernel_size_list[1]-1)-1)//self.stride_size_list[1])+1
        self.hw_out = self.hw_out//self.max_pool2d_size

        self.fc1 = nn.Linear(self.hw_out * self.hw_out * self.filters_list[-1], self.som_input_size)
        self.device = device
        self.som = SOM(input_dim=self.som_input_size, device=self.device)
        self.som = self.som.to(self.device)

    def cnn_extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, self.max_pool2d_size, self.max_pool2d_size)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, self.max_pool2d_size, self.max_pool2d_size)
        x = x.view(-1, self.hw_out*self.hw_out*self.filters_list[-1])
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
