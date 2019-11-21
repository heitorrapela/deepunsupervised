# Author: Heitor Rapela Medeiros <hrm@cin.ufpe.br>.
import numpy as np
from torchvision.datasets import VisionDataset
import random


def create_color_dataset():
    targets = np.empty((0, 1), int)
    colors = np.empty((0, 3), float)
    colors = np.append(colors, np.array([[0, 0, 0]]), axis=0)
    targets = np.append(targets, np.array([[0]]), axis=0)
    colors = np.append(colors, np.array([[1, 1, 1]]), axis=0)
    targets = np.append(targets, np.array([[1]]), axis=0)
    for i in range(10):
        for j in range(2, 13):
            targets = np.append(targets, np.array([[j]]), axis=0)
        colors = np.append(colors, np.array([[0, 0, random.random()]]), axis=0)
        colors = np.append(colors, np.array([[0, random.random(), 0]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), 0, 0]]), axis=0)
        colors = np.append(colors, np.array([[1, 1, random.random()]]), axis=0)
        colors = np.append(colors, np.array([[1, random.random(), 1]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), 1, 1]]), axis=0)
        colors = np.append(colors, np.array([[0, random.random(), random.random()]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), random.random(), 0]]), axis=0)
        colors = np.append(colors, np.array([[1, random.random(), random.random()]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), random.random(), 1]]), axis=0)
        colors = np.append(colors, np.array([[random.random(), random.random(), random.random()]]), axis=0)
    return np.array(colors), np.array(targets)


class COLORS(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None):
        super(COLORS, self).__init__(root, transform=transform, target_transform=target_transform)
        self.data, self.targets = create_color_dataset()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        img = np.transpose(img[:, np.newaxis])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.float(), target

    def __len__(self):
        return len(self.data)
