#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import datetime
import random
import sys
sys.path.append('../')

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, GlobalAvg
from models.test import test_img
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from utils.prune_parameters import *
from numpy import random
import json

def save_models(w_locals, w_global, masks, iters, path):
    save_dir = f"{path}/models"
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(f"{save_dir}/iters.json"):
        with open(f"{save_dir}/iters.json", 'r') as file:
            data = json.load(file)
        if data["iters"] > iters:
            return
    for i in range(len(w_locals)):
        torch.save(w_locals[i], f"{save_dir}/{i}.pth")
    for i in range(len(w_locals)):
        torch.save(masks[i], f"{save_dir}/mask{i}.pth")
    torch.save(w_global, f"{save_dir}/global.pth")
    json_data = {'iters': iters}
    with open(f"{save_dir}/iters.json", 'w') as file:
        json.dump(json_data, file)

def load_models(w_locals, w_global, masks, iters, path):
    save_dir = f"{path}/models"
    with open(f"{save_dir}/iters.json", 'r') as file:
        data = json.load(file)
    if data["iters"] < iters:
        return iters, w_global
    w_global = torch.load(f"{save_dir}/global.pth")
    for i in range(len(w_locals)):
        w_locals[i] = torch.load(f"{save_dir}/{i}.pth")
    for i in range(len(w_locals)):
        masks[i] = torch.load(f"{save_dir}/mask{i}.pth")
    return data["iters"], w_global

def save_path(args):
    pruner = args.pruner
    sparsity = 1 - args.compression**(-1)
    p = args.p
    comp = ""
    if args.p == 1:
        p = "1"
    else:
        p *= 10
        p = int(p)
        p = f"0{p}"
    if sparsity == 0:
        sparsity = "0"
    else:
        sparsity *= 10
        sparsity = int(sparsity)
        sparsity = f"0{sparsity}"

    if args.compensation:
        comp = "comp"
    else:
        comp = "nocomp"
    path = f"../save/results/{pruner}/{pruner}-p{p}-spar{sparsity}-{comp}"
    return path

def get_successful_users(p, num_users):
    group1 = []
    group2 = []
    for i in range(0, num_users//2):
        if random.random() <= p:
            group1.append(i)
    for i in range(num_users//2, num_users):
        if random.random() <= p:
            group2.append(i)
    if len(group1) == 0:
        user = np.random.choice(range(num_users//2), 1, replace=False)
        group1.append(user[0])
    if len(group2) == 0:
        user = np.random.choice(range(num_users//2, num_users), 1, replace=False)
        group2.append(user[0])
    return group1 + group2


def similarity_score(u: int,v: int, w_locals, args) -> float:
    """
    Calculate the similarity score between two clients based on the Euclidean distance.

    Args:
        u (int): Index of the first client.
        v (int): Index of the second client.
        w_locals: The weights of each client

    Returns:
        float: The similarity score between the two clients based on the Euclidean distance.
    """
    # print('locals: ', len(w_locals), type(w_locals), w_locals[:2])
    w_u, w_v = w_locals[u], w_locals[v] 
    diference_vector = torch.tensor([]).to(args.device)
    for k in w_u.keys():
        # print('size k', k, w_u[k].size())
        diference_vector = torch.cat((diference_vector, torch.flatten(w_u[k]) - torch.flatten(w_v[k])), dim=0)
    distance = torch.norm(diference_vector)
    # print(u, v, 'distance', distance)
    # print('w_v', u, v, type(w_v), w_v.keys())
    # print('w_v layer 0',  type(w_v['layers.0.weight']), w_v['layers.0.weight'] )
    return -distance

def similarity_based_compensation(w_locals, users_received, args, sm):
    similarity_matrix = [i for i in range(len(w_locals))]
    for u in range(args.num_users):
        if u in users_received:
            continue

        # find the most similar received / updated user
        most_similar = users_received[0]
        for neighbor in users_received:
            # print('u, v, most_similar', u, neighbor, most_similar, similarity_score(u, neighbor, w_locals), similarity_score(u, most_similar, w_locals))
            similarity = similarity_score(u, neighbor, w_locals, args)
            sm[u][neighbor] = similarity.item() * -1
            sm[neighbor][u] = similarity.item() * -1
            if similarity > sm[u][most_similar]:
                most_similar = neighbor

        # update the similarity matrix
        similarity_matrix[u] = most_similar
    print(similarity_matrix)
    return similarity_matrix

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    print(torch.cuda.get_device_name(0))
    path = save_path(args)
    print(path)

    seeds = [0]
    x_vals = [i for i in range(args.epochs_start, args.epochs_end, args.epochs_step)]
    y_vals = {'acc': [], 'loss': []}
    sm = [[0 for _ in range(args.num_users)] for _ in range(args.num_users)]

    for seed in seeds:
        # set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # load dataset and split users
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.dataset == 'cifar':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            else:
                dict_users, train_set = cifar_noniid(dataset_train, args.num_users)
        else:
            exit('Error: unrecognized dataset')
        img_size = dataset_train[0][0].shape

        # build model
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
            #model = models.vgg16(weights = None)

            # Step 4: Modify last layer
            #num_classes = 10  # CIFAR-10 has 10 classes
            #model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
            #net_glob = model.to(args.device)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)
        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        elif args.model == 'resnet18' and args.dataset == 'cifar':
            model = models.resnet18()
            # Modify the last layer to have 10 output classes (CIFAR-10 has 10 classes)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            net_glob = model.to(args.device)
        else:
            exit('Error: unrecognized model')
        print(net_glob)
        net_glob.train()

        # copy weights
        w_glob = net_glob.state_dict()

        # training
        loss_train = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = None
        val_acc_list, net_list = [], []
        masks = [randomMask(net_glob, args.device, args.compression) for _ in range(args.num_users)]
        print('mask', len(masks), len(masks[0]), type(masks[0]), masks[0][0].size()) # Shape = 100 x 6 x 400 x 3072
        w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]
        save_models(w_locals, w_glob, masks, 0, path)
        start_epochs, w_g = load_models(w_locals, w_glob, masks, 0, path)
        net_glob.load_state_dict(w_g)
        net_glob.eval()
        for iter in range(start_epochs, args.epochs_end):
            loss_locals = []
            idxs_users = get_successful_users(args.p, args.num_users) 
            print('RANDOM USER INDICES', idxs_users)
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss, mask = local.train(net=copy.deepcopy(net_glob).to(args.device), mask=masks[idx], train_iter=iter)
                # print(w.keys())
                masks[idx] = mask
                w_locals[idx] = copy.deepcopy(w)
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            # similarity logic
            if args.compensation:
                similarity_matrix = similarity_based_compensation(w_locals, users_received=idxs_users, args=args, sm = sm)
                w_glob = GlobalAvg(w_locals, similarity_matrix)
            else:
                w_glob = GlobalAvg(w_locals, idxs_users)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
            if (iter - args.epochs_start) % args.epochs_step == 0 and iter >= args.epochs_start:
                net_glob.eval()
                acc_test, loss_test = test_img(net_glob, dataset_train, args, train_set)
                print("Testing accuracy: {:.2f}".format(acc_test))
                print("Testing loss: {:.2f}".format(loss_test))

                y_vals['acc'].append(acc_test.item())
                y_vals['loss'].append(loss_test)
                #y_vals[args.pruner].append(0)
                save_models(w_locals, w_glob, masks, iter, path)

    print('test accuracy: ', y_vals['acc'])
    print('test loss: ', y_vals['loss'])
    # Plot both charts on the same axis
    plt.figure()
    plt.plot(x_vals, y_vals['acc'])
    plt.xlabel('Communication Rounds')
    plt.ylabel('Top-1 Accuracy')
    plt.title('')

    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    time = now.strftime("%H_%M_%S")

    # Create the directory if it doesn't exist
    save_dir = f"{path}/{date}"
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    plt.savefig(f"{save_dir}/similarity_test_acc_{args.pruner}_{args.prune_epochs}_{args.compensation}_{args.dataset}_{args.model}_{args.iid}_{args.p}_{args.num_users}_{args.epochs_start}_{args.epochs_end}_{time}.png")
    # plt.savefig('../save/synflow_test_{}_{}_{}_{}_{}_{}_{}.png'.format(args.prune_epochs, args.dataset, args.model, args.iid, args.frac, args.num_users, args.epochs))
    plt.figure()
    plt.plot(x_vals, y_vals['loss'])
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.title('')
    plt.savefig(f"{save_dir}/similarity_test_loss_{args.pruner}_{args.prune_epochs}_{args.compensation}_{args.dataset}_{args.model}_{args.iid}_{args.p}_{args.num_users}_{args.epochs_start}_{args.epochs_end}_{time}.png")
    np.savez(f"{save_dir}/similarity_test_loss_{args.pruner}_{args.prune_epochs}_{args.compensation}_{args.dataset}_{args.model}_{args.iid}_{args.p}_{args.num_users}_{args.epochs_start}_{args.epochs_end}_{time}", y = np.array(y_vals['loss']), x = np.array(x_vals))
    np.savez(f"{save_dir}/similarity_test_acc_{args.pruner}_{args.prune_epochs}_{args.compensation}_{args.dataset}_{args.model}_{args.iid}_{args.p}_{args.num_users}_{args.epochs_start}_{args.epochs_end}_{time}", y = np.array(y_vals['acc']), x = np.array(x_vals))
    plt.imshow(np.array(sm), cmap='viridis')  # cmap specifies the colormap (color scheme)
    plt.colorbar() 
    plt.savefig(f"{save_dir}/sm_{args.pruner}_{args.prune_epochs}_{args.compensation}_{args.dataset}_{args.model}_{args.iid}_{args.p}_{args.num_users}_{args.epochs_start}_{args.epochs_end}_{time}.png")