# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch
import pandas as pd

from scipy.io import arff
from sklearn import preprocessing
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from os.path import join
from .coil20 import COIL20
from .colors import COLORS


class Datasets(data.Dataset):

    def __init__(self, dataset, root_folder="raw-datasets/", flatten=False, coil20_unprocessed=False,
                 debug=False, n_samples=100):
        super(Datasets, self).__init__()

        transform_train_list = []
        transform_test_list = []

        if dataset == "mnist":
            transform_train_list.append(transforms.ToTensor())
            transform_train_list.append(transforms.Normalize((0.1307,), (0.3081,)))

            if flatten:
                transform_train_list.append(ReshapeTransform((-1,)))

            transform_train = transforms.Compose(transform_train_list)
            transform_test = transform_train

            self.train_data = datasets.MNIST(root=root_folder, train=True, download=True, transform=transform_train)
            self.test_data = datasets.MNIST(root=root_folder, train=False, download=True, transform=transform_test)
            
            if debug:
                self.train_data.data = self.train_data.data[:n_samples]
                self.test_data.data = self.test_data.data[:n_samples]

            self.dim_flatten = self.train_data.data.size(1) * self.train_data.data.size(2)

            self.d_in = 1
            self.hw_in = 28

        elif dataset == "fashion":
            transform_train_list.append(transforms.ToTensor())
            transform_train_list.append(transforms.Normalize((0.5,), (0.5,)))

            if flatten:
                transform_train_list.append(ReshapeTransform((-1,)))

            transform_train = transforms.Compose(transform_train_list)
            transform_test = transform_train

            self.train_data = datasets.FashionMNIST(root=root_folder, train=True,
                                                    download=True, transform=transform_train)
            self.test_data = datasets.FashionMNIST(root=root_folder, train=False,
                                                   download=True, transform=transform_test)

            if debug:
                self.train_data.data = self.train_data.data[:n_samples]
                self.test_data.data = self.test_data.data[:n_samples]

            self.dim_flatten = self.train_data.data.size(1) * self.train_data.data.size(2)

            self.d_in = 1
            self.hw_in = 28

        elif dataset == "cifar10":

            transform_train_list.append(transforms.RandomCrop(32, padding=4))
            transform_train_list.append(transforms.RandomHorizontalFlip())
            transform_train_list.append(transforms.ToTensor())
            transform_train_list.append(transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]))

            transform_test_list.append(transforms.ToTensor())
            transform_test_list.append(transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]))

            if flatten:
                transform_train_list.append(ReshapeTransform((-1,)))
                transform_test_list.append(ReshapeTransform((-1,)))

            transform_train = transforms.Compose(transform_train_list)
            transform_test = transforms.Compose(transform_test_list)

            self.train_data = datasets.CIFAR10(root=root_folder, train=True, download=True, transform=transform_train)
            self.test_data = datasets.CIFAR10(root=root_folder, train=False, download=True, transform=transform_test)
            
            if debug:
                self.train_data.data = self.train_data.data[:n_samples]
                self.test_data.data = self.test_data.data[:n_samples]
            
            data_shape = self.train_data.data.shape
            self.dim_flatten = data_shape[1] * data_shape[2] * data_shape[3]
            self.d_in = 3
            self.hw_in = 32

        elif dataset == "cifar100":
            transform_train_list.append(transforms.RandomCrop(32, padding=4))
            transform_train_list.append(transforms.RandomHorizontalFlip())
            transform_train_list.append(transforms.ToTensor())
            transform_train_list.append(transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]))

            transform_test_list.append(transforms.ToTensor())
            transform_test_list.append(transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]))

            if flatten:
                transform_train_list.append(ReshapeTransform((-1,)))
                transform_test_list.append(ReshapeTransform((-1,)))

            transform_train = transforms.Compose(transform_train_list)
            transform_test = transforms.Compose(transform_test_list)

            self.train_data = datasets.CIFAR100(root=root_folder, train=True, download=True, transform=transform_train)
            self.test_data = datasets.CIFAR100(root=root_folder, train=False, download=True, transform=transform_test)

            if debug:
                self.train_data.data = self.train_data.data[:n_samples]
                self.test_data.data = self.test_data.data[:n_samples]
            
            data_shape = self.train_data.data.shape
            self.dim_flatten = data_shape[1] * data_shape[2] * data_shape[3]
            self.d_in = 3
            self.hw_in = 32

        elif dataset == "svhn":
            transform_train_list.append(transforms.ToTensor())
            if flatten:
                transform_train_list.append(ReshapeTransform((-1,)))

            transform_train = transforms.Compose(transform_train_list)
            transform_test = transform_train

            self.train_data = datasets.SVHN(root=root_folder, split='train', download=True, transform=transform_train)
            self.test_data = datasets.SVHN(root=root_folder, split='test', download=True, transform=transform_test)

            if debug:
                self.train_data.data = self.train_data.data[:n_samples]
                self.test_data.data = self.test_data.data[:n_samples]

            data_shape = self.train_data.data.shape
            self.dim_flatten = data_shape[1] * data_shape[2] * data_shape[3]
            self.d_in = 3
            self.hw_in = 32

        elif dataset == "usps":

            transform_train_list.append(transforms.ToTensor())
            if flatten:
                transform_train_list.append(ReshapeTransform((-1,)))

            transform_train = transforms.Compose(transform_train_list)
            transform_test = transform_train

            self.train_data = datasets.USPS(root=root_folder, train=True, download=True, transform=transform_train)
            self.test_data = datasets.USPS(root=root_folder, train=False, download=True, transform=transform_test)

            if debug:
                self.train_data.data = self.train_data.data[:n_samples]
                self.test_data.data = self.test_data.data[:n_samples]

            data_shape = self.train_data.data.shape
            self.dim_flatten = data_shape[1] * data_shape[2]
            self.d_in = 1
            self.hw_in = 16

        elif dataset == "coil20":
            transform_train_list.append(transforms.ToTensor())

            if flatten:
                transform_train_list.append(ReshapeTransform((-1,)))

            transform_train = transforms.Compose(transform_train_list)
            transform_test = transform_train

            self.train_data = COIL20(root=root_folder, processed=not coil20_unprocessed,
                                     download=True, transform=transform_train)
            self.test_data = COIL20(root=root_folder, processed=not coil20_unprocessed,
                                    download=True, transform=transform_test)

            if debug:
                self.train_data.data = self.train_data.data[:n_samples]
                self.test_data.data = self.test_data.data[:n_samples]

            data_shape = self.train_data.data.shape
            self.dim_flatten = data_shape[1] * data_shape[2]
            self.d_in = 1
            self.hw_in = 32

        elif dataset == "colors":
            transform_train_list.append(transforms.ToTensor())

            if flatten:
                transform_train_list.append(ReshapeTransform((-1,)))

            transform_train = transforms.Compose(transform_train_list)
            transform_test = transform_train

            self.train_data = COLORS(root=root_folder, transform=transform_train)
            self.test_data = COLORS(root=root_folder, transform=transform_test)

            data_shape = self.train_data.data.shape
            self.dim_flatten = data_shape[1]
            self.d_in = 3
            self.hw_in = 1
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
