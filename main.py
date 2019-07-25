from Models.som import SOM
import numpy as np


data = np.ones((28*28*1))

som = SOM(input_size=data.shape[0], out_size=(28,28))