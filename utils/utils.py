import os
import pandas as pd
from scipy.io import arff
from os.path import join
import numpy as np


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
        targets = data.iloc[:, -1].values.astype('int16') if target_idx is None else data.ix[:, target_idx].values.astype('int16')

    return targets

