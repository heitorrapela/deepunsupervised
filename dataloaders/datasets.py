# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch
import pandas as pd

from scipy.io import arff
from sklearn import preprocessing
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from os.path import join


class Datasets(data.Dataset):

    def __init__(self, dataset, root_folder="raw-datasets/"):
        super(Datasets, self).__init__()

        transform = transforms.Compose([transforms.ToTensor(),
                                        ReshapeTransform((-1,))])

        if dataset == "mnist":
            self.train_data = datasets.MNIST(root=root_folder, train=True, download=True, transform=transform)
            self.test_data = datasets.MNIST(root=root_folder, train=False, download=True, transfsorm=transform)
            self.dim_flatten = self.train_data.data.size(1) * self.train_data.data.size(2)

        elif dataset == "fashion":
            self.train_data = datasets.FashionMNIST(root=root_folder, train=True, download=True, transform=transform)
            self.test_data = datasets.FashionMNIST(root=root_folder, train=False, download=True, transform=transform)
            self.dim_flatten = self.train_data.data.size(1) * self.train_data.data.size(2)

        elif dataset == "cifar10":
            self.train_data = datasets.CIFAR10(root=root_folder, train=True, download=True, transform=transform)
            self.test_data = datasets.CIFAR10(root=root_folder, train=False, download=True, transform=transform)
            data_shape = self.train_data.data.shape
            self.dim_flatten = data_shape[1] * data_shape[2] * data_shape[3]

        elif dataset == "cifar100":
            self.train_data = datasets.CIFAR100(root=root_folder, train=True, download=True, transform=transform)
            self.test_data = datasets.CIFAR100(root=root_folder, train=False, download=True, transform=transform)
            data_shape = self.train_data.data.shape
            self.dim_flatten = data_shape[1] * data_shape[2] * data_shape[3]

        elif dataset == "svhn":
            self.train_data = datasets.SVHN(root=root_folder, split='train', download=True, transform=transform)
            self.test_data = datasets.SVHN(root=root_folder, split='test', download=True, transform=transform)
            data_shape = self.train_data.data.shape
            self.dim_flatten = data_shape[1] * data_shape[2] * data_shape[3]

        else:
            self.train_data = CustomDataset(load_path=join(root_folder, dataset), norm="minmax")
            self.test_data = self.train_data
            self.dim_flatten = self.train_data.data.shape[1]


class CustomDataset(data.Dataset):

    def __init__(self, load_path, norm=None):
        super(CustomDataset, self).__init__()

        if load_path.endswith(".arff"):
            loaded_mat, meta = arff.loadarff(load_path)
            loaded_mat = pd.DataFrame(loaded_mat, dtype=float)
        else:
            loaded_mat = pd.read_csv(load_path, sep=",", header=None)
            loaded_mat = pd.DataFrame(loaded_mat, dtype=float)

        self.labels = loaded_mat.iloc[:, -1].values

        if norm is not None:
            if norm == 'minmax':
                min_max_scaler = preprocessing.MinMaxScaler().fit(loaded_mat)
                loaded_mat = min_max_scaler.transform(loaded_mat)

            elif norm == 'scaler':
                scaler = preprocessing.StandardScaler().fit(loaded_mat)
                loaded_mat = scaler.transform(loaded_mat)

        loaded_mat = pd.DataFrame(loaded_mat, dtype=float)

        self.data = loaded_mat.iloc[:, :-1]

        if load_path != "":
            self.labels = self.labels.astype(int)

    def __getitem__(self, index):
        sample, target = self.data.iloc[index], int(self.labels[index])

        return torch.tensor(sample), torch.tensor(target)

    def __len__(self):
        return len(self.data)


class ReshapeTransform(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        return torch.reshape(sample, self.size)