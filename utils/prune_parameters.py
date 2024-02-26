from torch import nn
def prunable(p):
    return isinstance(p, (nn.Linear, nn.Conv2d))
def parameters(model):
    for module in filter(lambda p: prunable(p), model.modules()):
        for param in module.parameters(recurse=False):
            if param is not module.bias:
                yield param