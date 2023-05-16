# sample ML script using PyTorch
# resources provided by Jon Krohn, https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/regression-in-pytorch.ipynb

import torch as t
import matplotlib.pyplot as plt

# set up x input for the model (graph) 
x_input = t.tensor([i for i in range(8)])
