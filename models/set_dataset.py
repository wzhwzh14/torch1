import numpy as np
import torch

xy=np.loadtxt('diabletes.csv',delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])