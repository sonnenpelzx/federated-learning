from torch import nn
import torch

def prunable(p):
    return isinstance(p, (nn.Linear, nn.Conv2d))

def parameters(model, device):
    filteredModel = list(filter(lambda p: prunable(p), model.modules()))
    for m_index in range(len(filteredModel)):
        if m_index == 0:
            continue
        module = filteredModel[m_index]
        for param in module.parameters(recurse=False):
            #if param is not module.bias:
            mask = torch.ones(param.shape).to(device)
            yield mask, param

def randomMask(model, device, compression):
    sparsity = 1.0 - (compression**(-1))
    masks = []
    filteredModel = list(filter(lambda p: prunable(p), model.modules()))
    for m_index in range(len(filteredModel)):
        if m_index == 0:
            continue
        module = filteredModel[m_index]
        for param in module.parameters(recurse=False):
            if len(param.shape) == 1 or param is module.bias:
                masks.append(torch.ones(param.shape).to(device))
                continue
            parameters_to_pruned = int(sparsity*(1-(param.shape[0]+param.shape[1])/(param.shape[0]*param.shape[1]))*param.numel())
            if isinstance(module, (nn.Conv2d)):
                w,h = module.kernel_size
                parameters_to_pruned = int(sparsity*(1-(param.shape[0]+param.shape[1]+w+h)/(param.shape[0]*param.shape[1]*w*h))*param.numel())
            m = torch.ones(param.shape)
            indices = torch.randperm(m.numel())[:parameters_to_pruned]
            m.view(-1)[indices] = 0
            m = m.to(device)
            masks.append(m)
    
    return masks