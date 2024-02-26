from utils.prune_parameters import *
import torch
import numpy as np

def prune(pruner, sparsity, device, epochs, model):
    for epoch in range(epochs):
        pruner.score(model, device)
        pruner.mask(model, sparsity, device)
    remaining_params, total_params = 0, 0
    for p in parameters(model): 
        zero = torch.tensor([0.]).to(device)
        one = torch.tensor([1.]).to(device)
        mask = torch.where(p == 0, zero, one)
        remaining_params += mask.detach().cpu().numpy().sum()
        total_params += p.numel()
    if np.abs(remaining_params - total_params * sparsity) >= 5:
        print(remaining_params, total_params, total_params*sparsity)