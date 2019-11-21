# Author: Heitor Rapela Medeiros <hrm@cin.ufpe.br>.
# ffmpeg -framerate 2 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4

from PIL import Image
from models.som import SOM
import torch
from sampling.custom_lhs import *
from argument_parser import argument_parser
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Argument Parser
    args = argument_parser()

    # Init the SOM
    som_size = 100
    batch_size = 32
    epochs = 200
    image_name = "./chamaleon.jpg"

    som = SOM(input_dim=3, n_max=som_size)

    img_original = Image.open(image_name)
    img_original = img_original.convert('RGB')
    img_input_matrix = np.asarray(img_original, dtype=np.float32)
    img_rows = img_input_matrix.shape[0]
    img_cols = img_input_matrix.shape[1]

    img_output_matrix = np.zeros((img_rows, img_cols, 3))

    for epoch in range(epochs):
        # Iterates through the original image and find the BMU for each single pixel.
        for row in range(img_rows):
            for col in range(img_cols):
                # print(row, img_rows, col, img_cols)
                input_vector = np.array(img_input_matrix[row, col, :] / 255.0)
                input_vector = torch.from_numpy(np.transpose(input_vector[:, np.newaxis])).float()
                _, bmu_weights, _ = som(input_vector)
                _, bmu_indexes = som.get_winners(input_vector)
                ind_max = bmu_indexes.item()
                img_output_matrix[row, col, :] = som.weights[ind_max]
            plt.imshow(img_output_matrix)
            plt.pause(0.00001)
