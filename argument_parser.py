# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description='Self Organizing Map')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--workers', type=int,  default=0, help='number of data loading workers')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--log-interval', type=int, default=32, help='Log Interval')
    parser.add_argument('--eval', action='store_true', help='enables evaluation')
    parser.add_argument('--eval-interval', type=int, default=32, help='Evaluation Interval')

    parser.add_argument('--disable-tensorboard', action='store_true', help='Disables Tensorboard')
    parser.add_argument('--tensorboard-root', type=str, default='tensorboard/', help='Tensorboard Root folder')

    parser.add_argument('--root', type=str, default='raw-datasets/', help='Dataset Root folder')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset Name')
    parser.add_argument('--out-folder', type=str, default='results/', help='Folder to output results')
    parser.add_argument('--batch-size', type=int, default=2, help='input batch size')

    parser.add_argument('--input-paths', default=None, help='Input Paths')
    parser.add_argument('--nmax', type=int, default=None, help='number of nodes')

    parser.add_argument('--lhs', action='store_true', help='enables lhs sampling before run')
    parser.add_argument('--lhs-samples', type=int, default=250, help='Number of Sets to be Sampled using LHS')
    parser.add_argument('--params-file', default=None, help='Parameters')

    parser.add_argument('--som-only', action='store_true', help='Som-Only Mode')
    parser.add_argument('--debug', action='store_true', help='Enables debug mode')
    parser.add_argument('--coil20-unprocessed', action='store_true', help='Loads COIL-20 Unprocessed')
    parser.add_argument('--n-samples', type=int, default=100, help='Dataset Number of Samples')
    parser.add_argument('--lr-cnn', type=float, default=0.00001, help='Learning Rate of CNN Model')
    parser.add_argument('--print', action='store_true', help='Print time')

    return parser.parse_args()