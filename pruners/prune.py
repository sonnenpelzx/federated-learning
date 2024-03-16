from utils.prune_parameters import *
import torch
import numpy as np

def prune(pruner, compression, prune_epochs, net, input_dim):
    sparsity = 0.0
    print(compression)
    for prune_epoch in range(prune_epochs):
        pruner.score(net, input_dim)
        sparsity = 1.0 - (compression) ** (-(prune_epoch+1)/prune_epochs)
        #print(sparsity)
        pruner.mask(sparsity)
    pruner.prune()
    print("Layer Collapse if = 0")
    pruner.score(net, input_dim, t=1)
    remaining_params, total_params = pruner.stats()
    # if np.abs(remaining_params - total_params * (1-sparsity)) >= 5:
    print(remaining_params, total_params, total_params*sparsity)