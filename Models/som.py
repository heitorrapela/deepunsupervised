import torch
import torch.nn as nn
from torchvision.utils import save_image
import pandas as pd
import re

class SOM(nn.Module):
    def __init__(self, input_size, out_size=(10, 10), lr=0.3, sigma=None,use_cuda=True):
        '''

        :param input_size:
        :param out_size:
        :param lr:
        :param sigma:
        '''
        super(SOM, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.relevances = None
        self.batch_size = 0
        self.dim = input_size
        self.lr = lr
        if sigma is None:
            self.sigma = max(out_size) / 2
        else:
            self.sigma = float(sigma)

        self.weight = nn.Parameter(torch.randn(input_size, out_size[0] * out_size[1]), requires_grad=False)
        self.locations = nn.Parameter(torch.Tensor(list(self.get_map_index())), requires_grad=False)
        self.pdist_fn = nn.PairwiseDistance(p=2)

    def get_map_index(self):
        '''Two-dimensional mapping function'''
        for x in range(self.out_size[0]):
            for y in range(self.out_size[1]):
                yield (x, y)

    def _neighborhood_fn(self, input, current_sigma):
        '''e^(-(input / sigma^2))'''
        input.div_(current_sigma ** 2)
        input.neg_()
        input.exp_()

        return input

    def forward(self, input):
        '''
        Find the location of best matching unit.
        :param input: data
        :return: location of best matching unit, loss
        '''
        self.batch_size = input.size()[0]
        input = input.view(self.batch_size, -1, 1)
        batch_weight = self.weight.expand(self.batch_size, -1, -1)

        dists = self.pdist_fn(input, batch_weight)
        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)
        bmu_locations = self.locations[bmu_indexes]

        return losses.sum().div_(self.batch_size).item(), bmu_indexes

    def self_organizing(self, input, current_iter, max_iter):
        '''
        Train the Self Oranizing Map(SOM)
        :param input: training data
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss (minimum distance)
        '''
        self.batch_size = input.size()[0]
        #Set learning rate
        iter_correction = 1.0 - current_iter / max_iter
        lr = self.lr * iter_correction
        sigma = self.sigma * iter_correction

        #Find best matching unit
        loss, _ = self.forward(input)

        # distance_squares = self.locations.float() - bmu_locations.float()
        # distance_squares.pow_(2)
        # distance_squares = torch.sum(distance_squares, dim=2)

        # lr_locations = self._neighborhood_fn(distance_squares, sigma)
        # lr_locations.mul_(lr).unsqueeze_(1)

        delta = lr * (input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0)
        delta.div_(self.batch_size)
        self.weight.data.add_(delta)

        return loss

    def save_result(self, dir, im_size=(0, 0, 0)):
        '''
        Visualizes the weight of the Self Oranizing Map(SOM)
        :param dir: directory to save
        :param im_size: (channels, size x, size y)
        :return:
        '''
        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1])

        images = images.permute(3, 0, 1, 2)
        save_image(images, dir, normalize=True, padding=1, nrow=self.out_size[0])

    def write_output(self, output_path, result):
        output_file = open(output_path, 'w+')

        content = str(self.weight.size(0)) + "\t" + str(self.weight.size(1)) + "\n"
        self.relevances = torch.ones(self.batch_size, self.dim, device=self.device)
        for i, relevance in enumerate(self.relevances.cpu()):
            content += str(i) + "\t" + "\t".join(map(str, relevance.numpy())) + "\n"

        result_text = result.to_string(header=False, index=False)
        result_text = re.sub('\n +', '\n', result_text)
        result_text = re.sub(' +', '\t', result_text)

        content += result_text
        output_file.write(content)
        output_file.close()

    def cluster(self, dataloader):
        clustering = pd.DataFrame(columns=['sample_ind', 'cluster'])
        for batch_idx, (inputs, targets) in enumerate(dataloader):

            _, bmu_indexes =  self.forward(inputs.to(self.device))
            ind_max = bmu_indexes.item()

            clustering = clustering.append({'sample_ind': batch_idx,
                                            'cluster': ind_max},
                                           ignore_index=True)

        return clustering
