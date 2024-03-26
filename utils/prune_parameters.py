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
    masks = []
    #parameters_pruned = 0
    for module in filter(lambda p: prunable(p), model.modules()):
        for param in module.parameters(recurse=False):
            parameters_to_pruned = int(sparsity*(1-(param.shape[0]+param.shape[1])/(param.shape[0]*param.shape[1]))*param.numel())
            m = torch.ones(param.shape)
            indices = torch.randperm(m.numel())[:parameters_to_pruned]
            m.view(-1)[indices] = 0
            m = m.to(device)
            if param is not module.bias:
                masks.append(m)
    #global_scores = torch.cat([torch.flatten(v) for v in masks])
    #print("sdfj")
    #print(global_scores.numel()*sparsity)
    #print(parameters_pruned)
    #k = int((sparsity) * global_scores.numel())
    #threshold = 0
    #if not k < 1:
    #    threshold, _ = torch.kthvalue(global_scores, k)
    #else:
    #    threshold = -1
    #for m in masks:
    #    zero = torch.tensor([0]).to(device)
    #    one = torch.tensor([1.]).to(device)
    #    m.copy_(torch.where(m <= threshold, zero, one))
    
    return masks