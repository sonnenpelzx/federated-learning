#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from pruners.prune import *
from pruners import pruners
import copy
from utils.prune_parameters import *

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
    
    def prune(self, net, mask):
        net.eval()
        (data, _) = next(iter(self.ldr_train))
        prune_methods = {
                'mag' : pruners.Mag,
                'synflow' : pruners.SynFlow,
                'fedspa': pruners.FedSpa,
            }
        input_dim = list(data[0,:].shape)
        print(self.args.pruner)
        pruner = prune_methods[self.args.pruner](net, self.args.device)
        if(self.args.pruner == 'fedspa'):
            pruner.use_mask(net, mask)
            remaining_params, total_params = pruner.stats()
            # if np.abs(remaining_params - total_params * (1-sparsity)) >= 5:
            print(remaining_params, total_params, total_params*(1-self.args.compression**(-1)),(1-self.args.compression**(-1)))
        else:
            prune(pruner, self.args.compression, self.args.prune_epochs, net, input_dim)
        net.train()
        return mask
    def test(self, net):
        net.zero_grad()
        (data, _) = next(iter(self.ldr_train))
        input_dim = list(data[0,:].shape)
        input1 = torch.ones([1] + input_dim).to(self.args.device)#, dtype=torch.float64).to(device)
        output1 = net(input1)
        input2 = torch.mul(input1, 2)
        output2 = net(input2)
        print(np.sum((output1).clone().detach().cpu().numpy()))
        #print(np.sum((output2).clone().detach().cpu().numpy()))
        #print(torch.eq(output1, output2))
        #print(output1)
        #print(output2)
        net.zero_grad()

    def train(self, net, mask):
        net.train()
        # train and update
        mask = self.prune(net, mask)
        #self.test(net)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), mask

