import json

from collections import defaultdict, OrderedDict
import math
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm_notebook as tqdm

from Accuracy import Accuracy
from Callback import Callback, CallbacksGroup
from datasets import dataset_getter
from logger import StreamLogger
from net import Net
import Parameters_schedule
from Parameters_schedule import Scheduler, OneCycleSchedule
from phase import Phase
from train import train
from utils import ProgressBar
from loss import RollingLoss
from datasets import dataset_getter as ds


def make_phases(train, valid, bs=128, n_jobs=0):
    return [
        Phase('train', DataLoader(train, bs, shuffle=True, num_workers=n_jobs)),
        Phase('valid', DataLoader(valid, bs, num_workers=n_jobs), grad=False)
    ]


def main2():
    default_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #### Get json file for experiment
    ##hyperparameters
    ###Load Hyperparameters
    data = 1
    data_path = Path.home()/'data'/data
    epochs = 3
    bs = 1
    n_jobs =1 
    lr = 1


    train_ds =1 

    valid_ds = 1
    

    phases = make_phases(train_ds, valid_ds, bs=bs, n_jobs=n_jobs)
    model = Net()
    opt = optim.Adam(model.parameters(), lr=lr)
    cb = CallbacksGroup([
        RollingLoss(),
        Accuracy(),
        Scheduler(
            OneCycleSchedule(t=len(phases[0].loader) * epochs),
            mode='batch'
        ),
        StreamLogger(),
        ProgressBar()
    ])



    train(model, opt, phases, cb, epochs=epochs, loss_fn=F.cross_entropy)


def main():
    default_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data_path = Path.home()/'data'/'mnist'

    mnist_stats = ((0.15,), (0.15,))

    epochs = 3

    train_ds = MNIST(
        data_path, 
        train=True, 
        download=True,
        transform=T.Compose([
            T.RandomAffine(5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(*mnist_stats)
        ])
    )

    valid_ds = MNIST(
        data_path, 
        train=False, 
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(*mnist_stats)
        ])
    )

    phases = make_phases(train_ds, valid_ds, bs=1024, n_jobs=4)
    model = Net()
    opt = optim.Adam(model.parameters(), lr=1e-2)
    cb = CallbacksGroup([
        RollingLoss(),
        Accuracy(),
        Scheduler(
            OneCycleSchedule(t=len(phases[0].loader) * epochs),
            mode='batch'
        ),
        StreamLogger(),
        ProgressBar()
    ])

    train(model, opt, phases, cb, epochs=epochs, device=default_device, loss_fn=F.cross_entropy)
    lr_history = pd.DataFrame(cb['scheduler'].parameter_history('lr'))
    ax = lr_history.plot(figsize=(8, 6))
    ax.set_xlabel('Training Batch Index')
    ax.set_ylabel('Learning Rate');



if __name__ == "__main__":
    main()