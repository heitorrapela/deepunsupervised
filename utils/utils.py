import os
import pandas as pd
from scipy.io import arff
from os.path import join
import numpy as np
import time


def read_params(file_path):
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path)
    else:
        data = []

    return data


def read_lines(file_path):
    if os.path.isfile(file_path):
        data = open(file_path, 'r')
        data = np.array(data.read().splitlines())
    else:
        data = []

    return data


def get_data_targets(path, file, target_idx=None):
    if file.endswith(".arff"):
        data, _ = arff.loadarff(open(join(path, file), 'rb'))
        targets = data['class'] if target_idx is None else data[target_idx]
    else:
        data = pd.read_csv(join(path, file), header=None)
        if target_idx is None:
            targets = data.iloc[:, -1].values.astype('int16')
        else:
            targets = data.ix[:, target_idx].values.astype('int16')

    return targets


class Timer(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time
        
    def tic(self):
        self.time = time.time()
        
    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count
