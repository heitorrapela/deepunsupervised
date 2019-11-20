# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import numpy as np

from torchvision.datasets.utils import download_url, makedir_exist_ok
from torchvision.datasets import VisionDataset

import os

from PIL import Image
import zipfile
import io
import re


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

