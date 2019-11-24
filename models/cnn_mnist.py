from __future__ import print_function
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.som import SOM

'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, d_in=1, n_conv_layers=3, batch_norm=False, max_pool=True, hw_in=28, som_input=2, filters_list=[20, 50],
                 kernel_size_list=[5, 5], stride_size_list=[1, 1], padding_size_list=[0, 0], max_pool2d_size=2,
                 n_max=20, at=0.985, eb=0.1, ds_beta=0.5, eps_ds=1., lp=0.05, device='cpu'):
        super(Net, self).__init__(),

        self.som_input_size = som_input
        self.d_in = d_in
        self.hw_out = hw_in
        self.max_pool = max_pool
        self.batch_norm = batch_norm
        self.n_conv_layers = n_conv_layers
        self.filters_list = [d_in] + list(np.power(2, self.generate_cnn_filters(filters_list)))
        self.max_pool2d_size = max_pool2d_size
        self.kernel_size_list = kernel_size_list
        self.padding_size_list = padding_size_list
        self.stride_size_list = stride_size_list
        self.convs = []
        self.device = device
        for i in range(self.n_conv_layers):
            if not (i < len(self.padding_size_list) and
                    i < len(self.kernel_size_list) and
                    i < len(self.stride_size_list)):
                print("Warning the size of the padding, kernel or stride list is too small!")
                break
            last_hw_out = ((self.hw_out + 2*self.padding_size_list[i]-(self.kernel_size_list[i]-1)-1)//self.stride_size_list[i])+1
            if last_hw_out >= 2 and last_hw_out%self.max_pool2d_size==0:
                self.hw_out = last_hw_out
                self.hw_out = self.hw_out//self.max_pool2d_size if self.max_pool else self.hw_out
                self.convs.append(nn.Sequential(nn.Conv2d(self.filters_list[i],
                                                          self.filters_list[i+1],
                                                          self.kernel_size_list[i],
                                                          self.stride_size_list[i],
                                                          self.padding_size_list[i]),
                                                nn.BatchNorm2d(self.filters_list[i+1]) if self.batch_norm else nn.Identity(),
                                                nn.ReLU(inplace=True),
                                                nn.MaxPool2d(self.max_pool2d_size,
                                                             self.max_pool2d_size) if self.max_pool else nn.Identity()).to(self.device))
            else:
                print("Warning the size of the output is too small!")
                break

        self.convs = nn.Sequential(*self.convs)
        self.fc1 = nn.Linear(self.hw_out*self.hw_out*self.filters_list[len(self.convs)], self.som_input_size).to(self.device)
        self.som = SOM(input_dim=self.som_input_size,
                       n_max=n_max,
                       eb=eb,
                       at=at,
                       ds_beta=ds_beta,
                       eps_ds=eps_ds,
                       lp=lp,
                       device=self.device)

        self.som = self.som.to(self.device)

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)


    def cnn_extract_features(self, x):
        x = self.convs(x)
        x = x.view(-1, self.hw_out*self.hw_out*self.filters_list[len(self.convs)])
        x = self.fc1(x)
        x = torch.tanh(x)
        return x

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = F.relu(self.fc1(out))
        out1 = F.relu(self.fc2(out))
        #out1 = F.sigmoid(out1)

        out = F.log_softmax(self.fc3(out1))
        #out1 = F.sigmoid(out1)

        return out, self.som(out1), out1#self.som(F.log_softmax(out1)), out1#self.som(self.cnn_extract_features(out1))
        #return
    
    def cluster(self, dataloader):
        clustering = pd.DataFrame(columns=['sample_ind', 'cluster'])
        predict_labels = []
        true_labels = []

        for batch_idx, (samples, targets) in enumerate(dataloader):
            samples, targets = samples.to(self.device), targets.to(self.device)
            x, _x1, outputs = self.forward(samples)

            #outputs = self.cnn_extract_features(samples)

            _, bmu_indexes = self.som.get_winners(outputs.to(self.device))

            for index, bmu_index in enumerate(bmu_indexes):
                ind_max = bmu_index.item()

                clustering = clustering.append({'sample_ind': batch_idx,
                                                'cluster': ind_max},
                                               ignore_index=True)
                predict_labels.append(ind_max)
                true_labels.append(targets[index].item())
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
