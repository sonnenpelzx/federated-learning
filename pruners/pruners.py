import torch    
import numpy as np
from utils.prune_parameters import *
import copy
import math

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

    def score(self, net, input_dim, t=0):
        for _, p in self.mask_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()
        
        #print R_SF = "sum of output when input is all ones and |W|"
        if t == 1:
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
            for i in range(len(masked_parameters_copy)):
                _, param = masked_parameters_copy[i]
                mask, _= self.mask_parameters[i]
                with torch.no_grad():
                    param.mul_(mask)
                param.requires_grad
            copy_net.zero_grad()
            input = torch.ones([1] + input_dim).to(self.device)#, dtype=torch.float64).to(device)
            output = copy_net(input)
            layer_collapse = np.sum((output).clone().detach().cpu().numpy())
            if layer_collapse == 0:
                print("layer collapse")

class FedSpa(Pruner):
    def __init__(self, net, device):
        super(FedSpa, self).__init__(net, device)

    def use_mask(self, net, input_dim, mask):
        for i in range(len(self.mask_parameters)):
            mask_p, param = self.mask_parameters[i]
            m = mask[i]
            mask_p.copy_(m)
            with torch.no_grad():
                param.mul_(m)
        
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
        for i in range(len(masked_parameters_copy)):
            _, param = masked_parameters_copy[i]
            mask, _= self.mask_parameters[i]
            with torch.no_grad():
                param.mul_(mask)
            param.requires_grad
        copy_net.zero_grad()
        input = torch.ones([1] + input_dim).to(self.device)#, dtype=torch.float64).to(device)
        output = copy_net(input)
        layer_collapse = np.sum((output).clone().detach().cpu().numpy())
        if layer_collapse == 0:
            print("layer collapse")
    def score(self, net, input_dim, t=0):
        for _, p in self.mask_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()
            
    def nextMask(self, net, mask, compression, training_iter, n_training_iter):

        def unravel_index(indices, shape):
            if len(shape) == 0:
                return ()
        
            unraveled_indices = []
            for s in reversed(shape):
                unraveled_indices.append(indices % s)
                indices //= s
        
            return tuple(reversed(unraveled_indices))
        
        sparsity =1- compression ** (-1)
        for i in range(len(self.mask_parameters)):
            mask_p, param = self.mask_parameters[i]
            m = mask[i]
            n = torch.numel(param)
            #define the number of additional parameters pruned via conise annialing
            k = int(0.05/2 * (1+ math.cos(training_iter * math.pi / n_training_iter))*(1-sparsity)*n)
            #prune the k parameters with the smallest absolute value
            before = m.detach().cpu().numpy().sum()
            if k > 0:
                remaining_params = int(m.detach().cpu().numpy().sum())
                if k >= remaining_params:
                    k = remaining_params
                score = torch.clone(param.data.mul(m)).detach().abs_().mul(-1)
                _, indeces1 = torch.topk(torch.flatten(score), k+n-remaining_params)
                unreavel_indeces = unravel_index(indeces1, m.shape)
                m[unreavel_indeces] = 0
                #grow the k parameters with the highest gradients
                zero = torch.tensor([0]).to(self.device)
                one = torch.tensor([1.]).to(self.device)
                m_inverse = torch.where(m == zero, one, zero)
                masked_gradients = param.grad.mul(m_inverse).add(m_inverse)
                _, indeces = torch.topk(torch.flatten(masked_gradients).abs(), k)
                unreavel_indeces = unravel_index(indeces, m.shape)
                m[unreavel_indeces] = 1
                after = m.detach().cpu().numpy().sum()
                if before - after != 0:
                    print("k", k, "before - after", before - after, "before", before, "after", after)
        return mask

class SynFlow(Pruner):
    def __init__(self, net, device):
        super(SynFlow, self).__init__(net, device)

    def score(self, net, input_dim, t = 0):
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
        mask_parameters_copy = list(parameters(copy_net, self.device))
        for i in range(len(mask_parameters_copy)):
            _, param = mask_parameters_copy[i]
            mask, _= self.mask_parameters[i]
            with torch.no_grad():
                param.mul_(mask)
            param.requires_grad
        copy_net.zero_grad()
        input = torch.ones([1] + input_dim).to(self.device)#, dtype=torch.float64).to(device)
        output = copy_net(input)
        torch.sum(output).backward()
        if np.sum((output).clone().detach().cpu().numpy()) == 0 and t == 1:
            print("layer collapse")
            t = 1   
        for i in range(len(mask_parameters_copy)):
            _, p = mask_parameters_copy[i]
            mask, param= self.mask_parameters[i]
            self.scores[id(param)] = torch.clone(p.grad * p).detach().abs_() 