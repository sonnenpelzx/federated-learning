import torch    
import numpy as np
from utils.prune_parameters import *

class Pruner:
    def __init__(self):
        self.scores = {}
    def score(self, model, device):
        raise NotImplementedError
    def mask(self, model, sparsity, device):
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for param in parameters(model):
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(device)
                one = torch.tensor([1.]).to(device)
                with torch.no_grad():
                    param *= torch.where(score <= threshold, zero, one)
                param.requires_grad
    
class Mag(Pruner):
    def __init__(self):
        super(Mag, self).__init__()

    def score(self, model, device):
        params = parameters(model)
        for p in params:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()
