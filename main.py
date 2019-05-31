import json

from collections import defaultdict, OrderedDict
import math
from pathlib import Path
import re
import sys

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import SubsetRandomSampler

from Accuracy import Accuracy
from Callback import Callback, CallbacksGroup
from datasets import HSI_Dataset
from logger import StreamLogger, CSVLogger
from net import LeeEtAl as Net
from saving import save_model
import Parameters_schedule
from Parameters_schedule import Scheduler, OneCycleSchedule
from phase import Phase
from train import train
from utils import ProgressBar
from utils import utils
from loss import RollingLoss


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import spectral.io.envi as envi
from spectral import *
from spectral import open_image as open_img


def make_phases(dataset, train, valid, bs=128, n_jobs=0):
    return [
        Phase('train', DataLoader(dataset, bs, sampler=train, num_workers=n_jobs)),
        Phase('valid', DataLoader(dataset, bs,
                                  sampler=valid, num_workers=n_jobs), grad=False)
    ]


def main():
    exp_path = Path.cwd()
    torch.cuda.empty_cache()
    # get user to choose training config
    config_file = "Lee_config.json"

    # Load Test Parameters
    with open(config_file, "r") as f:
        x = json.load(f)
    hyperparams = x["HYPERPARAMETERS"]
    name = x["model"]
    n_classes = hyperparams['n_classes']
    n_bands = hyperparams['n_bands']
    print(n_bands)
    patch_size = hyperparams['patch_size']
    _, path = utils.experiment_path_build(x)

    default_device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    HSI_ds = HSI_Dataset(path, transforms=None)
    # Params need moving
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    epochs = 2
    bs = 32
    n_jobs = 4
    lr = 1e-2

    # Creating data indices for training and validation splits:
    dataset_size = len(HSI_ds)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_ds_sampler = SubsetRandomSampler(train_indices)
    valid_ds_sampler = SubsetRandomSampler(val_indices)

    # train_ds = MNIST(
    #     data_path,
    #     train=True,
    #     download=True,
    #     transform=T.Compose([
    #         T.RandomAffine(5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    #         T.ToTensor(),
    #         T.Normalize(*mnist_stats)
    #     ])
    # )

    # valid_ds = MNIST(
    #     data_path,
    #     train=False,
    #     transform=T.Compose([
    #         T.ToTensor(),
    #         T.Normalize(*mnist_stats)
    #     ])
    # )

    phases = make_phases(HSI_ds, train_ds_sampler,
                         valid_ds_sampler, bs=32, n_jobs=4)
    model = Net(n_bands, n_classes, patch_size)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    training_params = {'Dataset Path': path,
                       'Experiment Path': exp_path,
                       'number of classes': n_classes,
                       'number of bands': n_bands,
                       'patch size': patch_size,
                       'test_train Split': {
                           'validation split': validation_split,
                           'shuffle dataset': shuffle_dataset,
                           'random seed': random_seed},
                       'Batch Size': bs,
                       "Nuber of Jobs": n_jobs,
                       "Learning Rate": lr,
                       "Scheduler": "One Cycle Schedule"
                       }

    cb = CallbacksGroup([
        RollingLoss(),
        Accuracy(),
        Scheduler(
            OneCycleSchedule(t=len(phases[0].loader) * epochs),
            mode='batch'
        ),
        StreamLogger(),
        CSVLogger(f'{path}/test.csv', training_params),
        ProgressBar(),
        save_model(exp_path, name)
    ])

    train(model, opt, phases, cb, epochs=epochs,
          device=default_device, loss_fn=F.cross_entropy)
    lr_history = pd.DataFrame(cb['scheduler'].parameter_history('lr'))
    ax = lr_history.plot(figsize=(8, 6))
    ax.set_xlabel('Training Batch Index')
    ax.set_ylabel('Learning Rate')
    fig = ax.get_figure()
    fig.savefig("lr-test.jpg")


if __name__ == "__main__":
    main()
