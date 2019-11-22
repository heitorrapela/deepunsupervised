# Author: Pedro Braga <phmb4@cin.ufpe.br>.

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


def read_results(results_file):
    results = open(results_file, 'r')
    # first line of results contains the number of found
    # clusters and the dimension of the data
    first_line = results.readline().split()
    found_clusters = int(first_line[0])
    dim = int(first_line[1])

    # finding found clusters id
    # data_n_winner is a tuple data id and winner id,
    # typical of .results files.
    data_n_winner = []
    for line in results:
        line_split = line.split()

        # results file contains a section with unnecessary data,
        # which takes more than two columns. we are interested in the
        # section with only two columns
        if len(line_split) == 2:
            data_n_winner.append(line_split)

    return data_n_winner, found_clusters, dim


def read_header(files, folder, header_rows, save_parameters=True):
    datasets = []
    folds = []
    headers = []

    for file in files:
        if ".csv" in file:
            header = pd.read_csv(join(folder, file), nrows=header_rows, header=None)
            header = header.transpose()
            header = header.rename(columns=header.iloc[0])
            header = header.drop([0])
            header = header.dropna(axis=0, how='any')
            header = header.astype(np.float64)

            headers.append(header)

            if len(datasets) <= 0:
                results = pd.read_csv(join(folder, file), skiprows=header_rows + 1, header=None)

                datasets = results.iloc[0]

                if 'lr_cnn' in datasets.values:
                    datasets = datasets[1: datasets[datasets == "lr_cnn"].index[0]]
                    if save_parameters:
                        save_params_file(results, "lr_cnn", folder)

                elif 'n_max' in datasets.values:
                    datasets = datasets[1: datasets[datasets == "n_max"].index[0]]
                    if save_parameters:
                        save_params_file(results, "n_max", folder)

                elif 'at' in datasets.values:
                    datasets = datasets[1: datasets[datasets == "at"].index[0]]
                    if save_parameters:
                        save_params_file(results, "at", folder)

                else:
                    datasets = datasets[1:]

    return datasets, folds, headers


def read_params_and_results(file_name, rows=5):
    results = pd.read_csv(file_name, skiprows=rows + 1, header=None)

    first_param_idx = results.iloc[0]

    if 'lr_cnn' in first_param_idx.values:
        first_param_idx = first_param_idx[first_param_idx == "lr_cnn"].index[0]
    elif 'n_max' in first_param_idx.values:
        first_param_idx = first_param_idx[first_param_idx == "n_max"].index[0]
    elif 'at' in first_param_idx.values:
        first_param_idx = first_param_idx[first_param_idx == "at"].index[0]
    else:
        first_param_idx = None

    if first_param_idx is not None:
        params = results.drop(results.columns[range(first_param_idx)], axis=1)
        params = params.rename(columns=params.iloc[0])
        params = params.drop([0])
        params = params.astype(np.float64)

        results = results.drop(results.columns[range(first_param_idx, len(results.columns))], axis=1)
        results = results.drop(results.columns[0], axis=1)
        results = results.rename(columns=results.iloc[0])
        results = results.drop([0])
    else:
        params = None
        results = None

    return params, results


def save_params_file(results, starting_param_name, filename):
    parameters = results.rename(columns=results.iloc[0])
    parameters = parameters.drop([0])
    parameters = parameters.astype(np.float64)
    parameters = parameters.iloc[:, parameters.columns.get_loc(starting_param_name):]

    min_row = parameters.min(0)
    max_row = parameters.max(0)
    min_max = pd.DataFrame([list(min_row), list(max_row)], columns=parameters.columns)

    full_data = min_max.append(parameters, ignore_index=True)

    first_column = map(str, range(len(parameters.index)))
    first_column.insert(0, 'max')
    first_column.insert(0, 'min')
    full_data.insert(0, '', first_column)

    if filename.endswith("/"):
        filename = filename[:-1]

    full_data.to_csv(join(filename, "parameters-" + filename + ".csv"), sep=',', index=False)


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
