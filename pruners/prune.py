from utils.prune_parameters import *
import torch
import numpy as np

def prune(pruner, compression, epochs, model, input_dimension):
    sparsity = 0.0
    for epoch in range(epochs):
        pruner.score(model, input_dimension)
        sparsity = 1.0 - (compression) ** (-(epoch+1)/epochs)
        pruner.mask(sparsity)
    pruner.prune()
    remaining_params, total_params = pruner.stats()
    if np.abs(remaining_params - total_params * (1-sparsity)) >= 5:
        print(remaining_params, total_params, total_params*sparsity)