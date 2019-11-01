# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import itertools
import munkres
import numpy as np
from scipy.io import arff


def permute_cols(a, inds):
    """
    Permutes the columns of matrix `a` given
    a list of tuples `inds` whose elements `(from, to)` describe how columns
    should be permuted.
    """

    p = np.zeros_like(a)
    for i in inds:
        p[i] = 1
    return np.dot(a, p)


def maximize_trace(a):
    """
    Maximize trace by minimizing the Frobenius norm of 
    `np.dot(p, a)-np.eye(a.shape[0])`, where `a` is square and
    `p` is a permutation matrix. Returns permuted version of `a` with
    maximal trace.
    """

    # Adding columns or rows with zeros to enforce that a is a square matrix.
    while a.shape[0] != a.shape[1]:

        if a.shape[0] < a.shape[1]:
            a = np.vstack((a, np.zeros(a.shape[1])))
        elif a.shape[1] < a.shape[0]:
            a = np.hstack((a, np.zeros((a.shape[0], 1))))

    assert a.shape[0] == a.shape[1]
    d = np.zeros_like(a)
    n = a.shape[0]
    b = np.eye(n, dtype=int)
    for i, j in itertools.product(range(n), range(n)):
        d[j, i] = sum((b[j, :]-a[i, :])**2)
    m = munkres.Munkres()
    inds = m.compute(d)
    return permute_cols(a, inds)


def clustering_error(confusion_matrix):

    """
    Calculates the CE (clustering error) of a clustering 
    represented by its confusion matrix.

    Note: disjoint clustering only, i.e., it only works for data that belong 
    to exactly one class label.
    """

    confusion_matrix = maximize_trace(confusion_matrix)

    ce = 1 - np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return ce


def results2confusion(data_file, results_file):

    """
    Receives the data file (in .arff format) and a .arff.results file, which contains clustering 
    assignments, and returns the confusion matrix 
    associated with the solution.

    If multiclass multilabel data (i.e., each instance can 
    belong to one or more categories), the label information in the data file must be in 
    one hot notation.

    For example: say the data is 3 dimensional and there are four possible 
    categories. A valid data file must be of the following format:

    3 1 2 0 0 1
    2 2 2 1 1 0
    4 5 1 0 1 1, and so on.

    The first data instance (array [3, 1, 2]) belongs solely to the 
    third label etc.

    In data file, label must be in the last column(s)
    
    """

    #import functions.read_arff
    #from . import read_arff

    #data = functions.read_arff.read_arff(data_file)     
    data = arff.loadarff(data_file)
    results = open(results_file, 'r')

    # first line of results contains the number of found 
    # clusters and the dimension of the data

    first_line = results.readline().split()
    qty_found_clusters = int(first_line[0])
    dim = int(first_line[1])

    # checking whether it's multiclass multilabel (subspace clustering) problem
    is_subspace = 0

    if data[0].shape[0] > dim + 1:

        is_subspace = 1

    # if single label
    if is_subspace == 0:

        # finding correct number of clusters and true clusters id's
        true_clusters_list = list(set(data[:,-1]))
        qty_true_clusters = len(true_clusters_list)

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
                data_n_winner.append(line.split()) 

        data_n_winner = np.asarray(data_n_winner, dtype=np.int)
        found_clusters_list = list(set(data_n_winner[:, -1]))

        # confusion matrix
        confusion = np.zeros((qty_found_clusters, qty_true_clusters))

        for i in range(data.shape[0]):
            curr_data = data[i]
            
            # row of the confusion matrix to be updated
            curr_winner = data_n_winner[i, 1]
            row = found_clusters_list.index(curr_winner)

            # column to be updated
            curr_true = curr_data[-1]
            column = true_clusters_list.index(curr_true)

            confusion[row, column] += 1 

    return confusion


def conditional_entropy(confusion_matrix):

    """
    Given a confusion matrix, computes the conditional entropy like in Tuytelaars 2008, 
    Unsupervised Object Discovery: a Comparison

    Obs.: the rows of the confusion matrix must be the found clusters, whereas 
    the columns are the reference (ground truth), i.e., the number of rows must equal 
    the number of found clusters, and the number of columns must 
    equal the number of class labels.
    """

    # array with probability of each found cluster 
    py = np.zeros(confusion_matrix.shape[0])

    total_sum = np.sum(confusion_matrix)

    for i in range(confusion_matrix.shape[0]):
        py[i] = np.sum(confusion_matrix[i]) / total_sum

    pxDy = np.zeros((confusion_matrix.shape[0], confusion_matrix.shape[1]))
    for i in range(confusion_matrix.shape[0]):
        row_sum = np.sum(confusion_matrix[i])+0.00001
        for j in range(confusion_matrix.shape[1]):
            pxDy[i, j] = confusion_matrix[i, j] / row_sum

    out = 0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            out += py[i] * pxDy[i, j] * np.log2(1/(pxDy[i, j]+0.0001))

    return out


def multilabelresults2clustering_error(data_file, results_file, qty_categories):

    """
    Receives .arff file, results file and quantity of categories. Returns the 
    confusion matrix when data instances may belong to one or more labels

    The labels, in reality, are transformed to decimal. Example: if original label 
    was [1, 0, 1], meaning that the instance belongs to first and third classes, 
    the label in the file is the decimal notation, i.e., 5.
    """

    #from . import read_arff

    data = arff.loadarff(data_file)
    results = open(results_file, 'r')

    # first line of results contains the number of found 
    # clusters and the dimension of the data

    first_line = results.readline().split()
    qty_found_clusters = int(first_line[0])
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
            data_n_winner.append(line.split()) 
    if len(data_n_winner) == 0:
        data_n_winner.append([0, 0])

    data_n_winner = np.asarray(data_n_winner, dtype=np.int)
    found_clusters_list = list(set(data_n_winner[:, -1]))

    # converting winners to binary array
    bin_winner_array = np.zeros((data.shape[0], qty_found_clusters))

    for cell in data_n_winner:

        data_id = cell[0]
        winner_id = cell[1]

        bin_winner_array[data_id, found_clusters_list.index(winner_id)] = 1


    # building bin label array
    bin_label_array = np.zeros((data.shape[0], qty_categories))

    for i in range(data.shape[0]):

        label = data[i, -1]
        binary_label = np.array(list(bin(int(label))[2:].zfill(int(qty_categories))))
        bin_label_array[i] = binary_label


    # now updating confusion matrix
    
    confusion_matrix = np.zeros((qty_found_clusters, qty_categories))
    for i in range(data.shape[0]):

        for j in range(bin_label_array.shape[1]):

            for k in range(bin_winner_array.shape[1]):

                if bin_label_array[i, j] == 1 and bin_winner_array[i, k] == 1:

                    confusion_matrix[k, j] += 1

    #confusion_matrix = confusion_matrix.T
    #print(confusion_matrix.shape)
    outconf = np.copy(confusion_matrix)
    confusion_matrix = maximize_trace(confusion_matrix)
    #print(outconf.shape)

    # finding union size of all data points
    count = 0

    for i in range(bin_winner_array.shape[0]):

        qty_non_zero1 = len(bin_label_array[i][bin_label_array[i] != 0])
        qty_non_zero2 = len(bin_winner_array[i][bin_winner_array[i] != 0])
        count += np.max([qty_non_zero1, qty_non_zero2])

    ce = 1-confusion_matrix.trace() / count

    return ce, outconf
