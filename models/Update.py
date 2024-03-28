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
        pruner = prune_methods[self.args.pruner](net, self.args.device)
        if(self.args.pruner == 'fedspa'):
            pruner.use_mask(net,input_dim, mask)
        else:
            prune(pruner, self.args.compression, self.args.prune_epochs, net, input_dim)
        net.train()
        return mask

    def train(self, net, mask, train_iter):
        net.train()
        mask = self.prune(net, mask)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if(len(images) == 1):
                    continue
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
        if(self.args.pruner == 'fedspa'):
            pruner = pruners.FedSpa(net, self.args.device)
            mask = pruner.nextMask(net, mask, self.args.compression, train_iter, self.args.epochs)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), mask

