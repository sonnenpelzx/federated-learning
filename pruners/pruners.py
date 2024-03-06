import torch    
import numpy as np
from utils.prune_parameters import *
import copy

class Pruner:
    def __init__(self, net, device):
        self.scores = {}
        self.device = device
        self.mask_parameters = list(parameters(net, device))
    def score(self, net, imput_dimension):
        raise NotImplementedError
    def mask(self, sparsity):
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            #print("threshold", threshold)
            for m, param in self.mask_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0]).to(self.device)
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
                param.mul_(mask)
class Mag(Pruner):
    def __init__(self, net, device):
        super(Mag, self).__init__(net, device)

    def score(self, net, input_dim):
        for _, p in self.mask_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()

class SynFlow(Pruner):
    def __init__(self, net, device):
        super(SynFlow, self).__init__(net, device)

    def score(self, net, input_dim):
        
        def test(net, input_dim):
            net.zero_grad()
            input1 = torch.ones([1] + input_dim).to(self.device)#, dtype=torch.float64).to(device)
            output1 = net(input1)
            net.zero_grad()
            input2 = torch.mul(input1, 2)
            output2 = net(input2)
            e = torch.eq(output1, output2).all().item()
            return e
        @torch.no_grad()
        def linearize(model):
            #model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        copy_net = copy.deepcopy(net)
        copy_net.eval()
        signs = linearize(copy_net)
        masked_parameters_copy = list(parameters(copy_net, self.device))
        t = 0
        for i in range(len(masked_parameters_copy)):
            _, param = masked_parameters_copy[i]
            mask, _= self.mask_parameters[i]
            with torch.no_grad():
                param.mul_(mask)
            param.requires_grad
            if test(copy_net, input_dim):
                t = 1
        remaining_params, total_params = self.stats()
        copy_net.zero_grad()
        input = torch.ones([1] + input_dim).to(self.device)#, dtype=torch.float64).to(device)
        output = copy_net(input)
        torch.sum(output).backward()
        for i in range(len(masked_parameters_copy)):
            _, p = masked_parameters_copy[i]
            mask, param= self.mask_parameters[i]
            if t == 1:
                print("layer", i)
                print(remaining_params, total_params)
                print()
            self.scores[id(param)] = torch.clone(p.grad * p).detach().abs_() 