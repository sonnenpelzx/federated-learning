import torch    
import numpy as np
from utils.prune_parameters import *
import copy

class Pruner:
    def __init__(self, model, device):
        self.scores = {}
        self.device = device
        self.mask_parameters = list(parameters(model, device))
    def score(self, model, imput_dimension):
        raise NotImplementedError
    def mask(self, sparsity):
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for m, param in self.mask_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(self.device)
                one = torch.tensor([1.]).to(self.device)
                m.copy_(torch.where(score <= threshold, zero, one))
    def stats(self):
        remaining_params, total_params = 0, 0
        for mask, p in self.mask_parameters: 
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += p.numel()
        return remaining_params, total_params 
    def prune(self):
        for mask, param in self.mask_parameters:
            with torch.no_grad():
                param *= mask
            param.requires_grad
class Mag(Pruner):
    def __init__(self, model, device):
        super(Mag, self).__init__(model, device)

    def score(self, model, input_dimension):
        for _, p in self.mask_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()

class SynFlow(Pruner):
    def __init__(self, model, device):
        super(SynFlow, self).__init__(model, device)

    def score(self, model, input_dimension):
      
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        net = copy.deepcopy(model)
        signs = linearize(net)
        net.zero_grad()
        input = torch.ones([1] + input_dimension).to(self.device)#, dtype=torch.float64).to(device)
        output = net(input)
        torch.sum(output).backward()
        masked_parameters = list(parameters(net, self.device))
        for i in range(len(masked_parameters)):
            _, p = masked_parameters[i]
            _, param = self.mask_parameters[i]
            self.scores[id(param)] = torch.clone(p.grad * p).detach().abs_()
        

        nonlinearize(model, signs)
