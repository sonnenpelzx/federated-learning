#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def GlobalAvg(w, similarity_matrix):
    similar_w = [w[i] for i in range(len(w))]
    for i in range(len(similarity_matrix)):
        similar_w[i] = w[similarity_matrix[i]]
    return FedAvg(similar_w)