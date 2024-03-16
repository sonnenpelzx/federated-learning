from torch import nn
import torch

def prunable(p):
    return isinstance(p, (nn.Linear, nn.Conv2d))

def parameters(model, device):
    for module in filter(lambda p: prunable(p), model.modules()):
        for param in module.parameters(recurse=False):
            if param is not module.bias:
                mask = torch.ones(param.shape).to(device)
                yield mask, param

def randomMask(model, device, compression):
    sparsity = 1.0 - (compression**(-1))
    print(sparsity)
    masks = []
    for module in filter(lambda p: prunable(p), model.modules()):
        for param in module.parameters(recurse=False):
            if param is not module.bias:
                masks.append(torch.rand(param.shape).to(device))
    global_scores = torch.cat([torch.flatten(v) for v in masks])
    k = int((sparsity) * global_scores.numel())
    threshold = 0
    if not k < 1:
        threshold, _ = torch.kthvalue(global_scores, k)
    else:
        threshold = -1
    for m in masks:
        zero = torch.tensor([0]).to(device)
        one = torch.tensor([1.]).to(device)
        m.copy_(torch.where(m <= threshold, zero, one))
    
    remaining_params = 0
    for mask in masks: 
        remaining_params += mask.detach().cpu().numpy().sum()
    print(remaining_params )
    return masks