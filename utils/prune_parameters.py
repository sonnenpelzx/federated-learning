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