# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch
import pandas as pd
import numpy as np

from scipy.io import arff
from sklearn import preprocessing
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.datasets.utils import download_url, makedir_exist_ok
from torchvision.datasets import VisionDataset

import os
from os.path import join

from PIL import Image
import zipfile
import io
import re


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


class COIL20(VisionDataset):

    type_list = {
        'processed': [
            "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
            "coil-20-proc.zip"
        ],
        'unprocessed': [
            "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip",
            "coil-20-unproc.zip"
        ]
    }

    def __init__(self, root, processed=True, transform=None, target_transform=None,
                 download=False):
        super(COIL20, self).__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        self.data, self.targets = self.process(processed)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        """Download the COIL20 data if it doesn't exist already."""
        # download files

        if self._check_exists():
            return

        makedir_exist_ok(self.unprocessed_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        url, filename = self.type_list['processed']
        download_url(url, root=self.processed_folder, filename=filename)

        url, filename = self.type_list['unprocessed']
        download_url(url, root=self.unprocessed_folder, filename=filename)

    def _check_exists(self):
        return os.path.exists(self.processed_folder) and os.path.exists(self.unprocessed_folder)

    def process(self, processed):
        # process files
        data_type = 'processed' if processed else 'unprocessed'
        root = self.processed_folder if processed else self.unprocessed_folder

        url, filename = self.type_list[data_type]

        samples = []
        targets = []
        archive_path = os.path.join(root, filename)
        with zipfile.ZipFile(archive_path, 'r') as z:
            for img_name in z.namelist():
                if re.search('obj', img_name) is not None:
                    raw_data = z.read(img_name)
                    byte_data = io.BytesIO(raw_data)
                    img = np.asarray(Image.open(byte_data))
                    samples.append(img)
                    targets.append(int(img_name.split("__")[0].split("/obj")[1]))

        return np.array(samples), np.array(targets)

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def unprocessed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'unprocessed')


class ReshapeTransform(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        return torch.reshape(sample, self.size)
