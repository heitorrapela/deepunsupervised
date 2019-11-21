import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 
from PIL import Image

import os

from models.som import SOM
import torch.backends.cudnn as cudnn
import random
from torch.utils.data.dataloader import DataLoader
import argparse
import metrics
from models.cnn_mnist import Net
import torch.optim as optim
import torch
import torch.nn as nn
from utils import utils
from utils.plot import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os.path import join
from sampling.custom_lhs import *
from cuml.manifold import TSNE as cumlTSNE

def argument_parser():
	parser = argparse.ArgumentParser(description='Self Organizing Map')
	parser.add_argument('--cuda', action='store_true', help='enables cuda')
	parser.add_argument('--workers', type=int,  default=0, help='number of data loading workers')
	parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
	parser.add_argument('--log-interval', type=int, default=32, help='Log Interval')
	parser.add_argument('--eval', action='store_true', help='enables evaluation')
	parser.add_argument('--eval-interval', type=int, default=32, help='Evaluation Interval')

	parser.add_argument('--root', type=str, default='raw-datasets/', help='Dataset Root folder')
	parser.add_argument('--tensorboard-root', type=str, default='tensorboard/', help='Tensorboard Root folder')
	parser.add_argument('--dataset', type=str, default='mnist', help='Dataset Name')
	parser.add_argument('--out-folder', type=str, default='results/', help='Folder to output results')
	parser.add_argument('--batch-size', type=int, default=1, help='input batch size')

	parser.add_argument('--input-paths', default=None, help='Input Paths')
	parser.add_argument('--nmax', type=int, default=None, help='number of nodes')

	parser.add_argument('--lhs', action='store_true', help='enables lhs sampling before run')
	parser.add_argument('--lhs-samples', type=int, default=250, help='Number of Sets to be Sampled using LHS')
	parser.add_argument('--params-file', default=None, help='Parameters')

	parser.add_argument('--som-only', action='store_true', help='Som-Only Mode')
	parser.add_argument('--debug', action='store_true', help='Enables debug mode')
	parser.add_argument('--n-samples', type=int, default=100, help='Dataset Number of Samples')
	parser.add_argument('--lr-cnn', type=float, default=0.00001, help='Learning Rate of CNN Model')

	return parser.parse_args()


if __name__ == '__main__':
	# Argument Parser
	args = argument_parser()

	if torch.cuda.is_available() and not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	out_folder = args.out_folder if args.out_folder.endswith("/") else args.out_folder + "/"
	if not os.path.exists(os.path.dirname(out_folder)):
		os.makedirs(os.path.dirname(out_folder), exist_ok=True)

	tensorboard_root = args.tensorboard_root
	if not os.path.exists(os.path.dirname(tensorboard_root)):
		os.makedirs(os.path.dirname(tensorboard_root), exist_ok=True)

	tensorboard_folder = join(tensorboard_root, datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
	writer = SummaryWriter(tensorboard_folder)
	print("tensorboard --logdir=" + tensorboard_folder)

	use_cuda = torch.cuda.is_available() and args.cuda

	if use_cuda:
		torch.cuda.init()

	device = torch.device('cuda:0' if use_cuda else 'cpu')

	ngpu = int(args.ngpu)

	root = args.root
	dataset_path = args.dataset
	batch_size = args.batch_size
	debug = args.debug
	n_samples = args.n_samples
	lr_cnn = args.lr_cnn

	input_paths = utils.read_lines(args.input_paths) if args.input_paths is not None else None
	n_max = args.nmax

	params_file_som = args.params_file if args.params_file is not None else "arguments/default_som.lhs"

	manual_seed = 1
	random.seed(manual_seed)
	torch.manual_seed(manual_seed)

	#Set to True if you want to save the SOM images inside a folder.
	SAVE_IMAGE = True
	output_path = "./output/" #Change this path to save in a different forlder
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	#Init the SOM
	som_size = 100
	batch_size = 32
	tot_epoch = 200
	image_name = "chameleon.jpg" #the file must be in the "./input" folder
	
	som = SOM(input_dim=3, n_max=som_size, device=device)
	#som_epochs = param_set.epochs

	output_path = "./output/" #Change this path to save in a different forlder
	if not os.path.exists(output_path):
		os.makedirs(output_path) 
	img_original = Image.open(image_name)
	img_input_matrix = np.asarray(img_original, dtype=np.float32)
	img_rows = img_input_matrix.shape[0]
	img_cols = img_input_matrix.shape[1]

	img_output_matrix = np.zeros((img_rows, img_cols, 3))
	#Starting the Learning


	for epoch in range(tot_epoch):
		
		#Iterates the elements in img_output_matrix and
		#assign the closest value contained in SOM
		

		#Iterates through the original image and find the BMU for
		#each single pixel.
		for row in range(img_rows):

			for col in range(img_cols):
				print(row,img_rows,col,img_cols)
				input_vector =  np.array(img_input_matrix[row, col, :]/255.0)
				input_vector = np.transpose(input_vector[:,np.newaxis])
				input_vector = torch.from_numpy(input_vector).float()
				#print(input_vector.shape)

				_ , bmu_weights, _ = som(input_vector)
				_, bmu_indexes = som.get_winners(input_vector)
				ind_max = bmu_indexes.item()
				#print()
				#bmu_index = my_som.return_BMU_index(input_vector)
				#bmu_weights = my_som.get_unit_weights(bmu_index[0], bmu_index[1])
				#print(bmu_weights)
				#print((som.weights[ind_max] - 255) * -1)
				img_output_matrix[row, col, :] = (som.weights[ind_max]) * 255.0 #renormalise to show the right colours 
				#print(epoch,tot_epoch,row,img_rows,col,img_cols)
				#if(row == 1 and col == 1):
				#	print(input_vector)
				#	print(img_output_matrix[row, col, :]) 
				#	print(som.weights[ind_max])         
				#input()#plt.pause(0.001)
		#print("oi")
		#plt.axis("off")
		#plt.imshow(img_output_matrix)
		#plt.pause(0.001)
		#plt.show()

		#Saving the final image
		#img_output = ((img_output_matrix- 255) * -1 ).astype(np.uint8)
		img_to_save = Image.fromarray(img_output_matrix.astype(np.uint8), "RGB")
		img_to_save.save(str(epoch) + '.jpg')

		#plt.pause(0.001)


		#return updatable_samples_hight_at, self.weights[unique_nodes_high_at], self.relevance[unique_nodes_high_at]

		#print(a.shape, bmu_weights.shape, c.shape)

		#print(bmu_weights.shape)
	#img = np.rint(som.weights.view(16,16,3)*255)
