from __future__ import print_function
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.som import SOM

class AutoEncoderSOM(nn.Module):
    def __init__(self, d_in=1, hw_in=28, som_input=2,
                 n_max=20, at=0.985, eb=0.1, ds_beta=0.5, eps_ds=1., ld=0.05, device='cpu'):
        super(AutoEncoderSOM, self).__init__(),

        self.som_input_size = 16*20*20 #som_input
        self.d_in = d_in
        self.hw_out = hw_in
        self.device = device

        self.som = SOM(input_dim=self.som_input_size,
                       n_max=n_max,
                       eb=eb,
                       at=at,
                       ds_beta=ds_beta,
                       eps_ds=eps_ds,
                       ld=ld,
                       device=self.device)

        self.som = self.som.to(self.device)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.d_in, 32, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())

    def cnn_extract_features(self, x):
        encoded_features = self.encoder(x)
        return torch.tanh(encoded_features).view(-1,self.som_input_size)


    def forward(self, x):
        encoded_features = self.encoder(x)
        decoded_features = encoded_features
        decoded_features = self.decoder(decoded_features)
        return decoded_features, self.som(torch.tanh(encoded_features).view(-1,self.som_input_size))#self.som(self.cnn_extract_features(x))
    
    def cluster(self, dataloader):
        clustering = pd.DataFrame(columns=['sample_ind', 'cluster'])
        predict_labels = []
        true_labels = []

        for batch_idx, (samples, targets) in enumerate(dataloader):
            samples, targets = samples.to(self.device), targets.to(self.device)
            outputs = self.cnn_extract_features(samples)

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
