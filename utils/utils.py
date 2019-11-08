import os
import pandas as pd
import re
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


def write_som_output(som_model, output_path, result):
    output_file = open(output_path, 'w+')

    n_clusters = som_model.node_control[som_model.node_control == 1].size(0)

    content = str(n_clusters) + "\t" + str(som_model.input_size) + "\n"
    for i, relevance in enumerate(som_model.relevance):
        if som_model.node_control[i] == 1:
            content += str(i) + "\t" + "\t".join(map(str, relevance.numpy())) + "\n"

    result_text = result.to_string(header=False, index=False).strip()
    result_text = re.sub('\n +', '\n', result_text)
    result_text = re.sub(' +', '\t', result_text)

    content += result_text
    output_file.write(content)
    output_file.close()
