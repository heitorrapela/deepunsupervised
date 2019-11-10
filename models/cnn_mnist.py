from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from torchvision import datasets, transforms
from models.som import SOM


class Net(nn.Module):
    def __init__(self, device='cpu', som_input=2,filters_list=[20,50],kernel_size_list=[5,5],padding_size_list=[1,1]):
        super(Net, self).__init__()

        self.som_input_size = som_input
        self.filters_list = filters_list
        self.kernel_size_list = kernel_size_list
        self.padding_size_list = padding_size_list
        self.conv1 = nn.Conv2d(1, self.filters_list[0], self.kernel_size_list[0], self.padding_size_list[0])
        self.conv2 = nn.Conv2d(self.filters_list[0], self.filters_list[1], self.kernel_size_list[1], self.padding_size_list[1])
        self.fc1 = nn.Linear(4 * 4 * self.filters_list[-1], self.som_input_size)
        self.device = device
        self.som = SOM(input_dim = self.som_input_size, device = self.device)
        self.som = self.som.to(self.device)

    def cnn_extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*self.filters_list[-1])
        x = self.fc1(x)
        x = torch.tanh(x)
        return x

    def forward(self, x):
        return self.som(self.cnn_extract_features(x))
    
    def cluster(self, dataloader):
        clustering = pd.DataFrame(columns=['sample_ind', 'cluster'])
        predict_labels = []
        true_labels = []

        for batch_idx, (inputs, targets) in enumerate(dataloader):

            outputs = self.cnn_extract_features(inputs.to(self.device))
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
