# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch
import pandas as pd

from scipy.io import arff
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from os.path import join


class Datasets(Dataset):

    def __init__(self, dataset, root_folder="Datasets/", batch_size=32):

        transform = transforms.Compose([transforms.ToTensor()])

        if dataset == "mnist":
            data = datasets.MNIST(root=root_folder, train=True, download=True, transform=transform)
            test_data = datasets.MNIST(root=root_folder, train=False, download=True, transform=transform)

        elif dataset == "fashion":
            data = datasets.FashionMNIST(root=root_folder, train=True, download=True, transform=transform)
            test_data = datasets.FashionMNIST(root=root_folder, train=False, download=True, transform=transform)
        else:
            data = CustomDataset(load_path=join(root_folder, dataset))
            test_data = data

        self.train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, shuffle=False)


class CustomDataset(Dataset):

    def __init__(self, load_path):

        if load_path.endswith(".arff"):
            data, meta = arff.loadarff(load_path)
            data = pd.DataFrame(data, dtype=float)
        else:
            data = pd.read_csv(load_path, sep=",", header=None)
            data = pd.DataFrame(data, dtype=float)

        self.y = data.iloc[:, -1].values

        self.X = data.iloc[:, :-1]

        if load_path != "":
            self.y = self.y.astype(int)

    def __getitem__(self, index):
        data = torch.tensor(self.X.iloc[index])
        target = torch.tensor(self.y[index], dtype=torch.long)
        return data, target

    def __len__(self):
        return self.X.shape[0]
