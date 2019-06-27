"""
Hyperspectral

"""

import json

from pathlib import Path

import pandas as pd
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader


from Accuracy import Accuracy
from Callback import CallbacksGroup
from datasets import HSI_Dataset
from logger import StreamLogger, CSVLogger
from net import ResNet as Net
from net import ResidualBlock
from saving import save_model
from Parameters_schedule import Scheduler, OneCycleSchedule
from phase import Phase
from train import train
from utils import ProgressBar
from utils import utils, train_test_valid_split
from loss import RollingLoss
from test import test_images


def make_phases(dataset, training, valid, test, bs=32, n_jobs=0, disp_batch=5):
    """
    make phases builds the dataloaders for each training phase and implements flags for some callback methods.
    """
    return [
        Phase('train', DataLoader(dataset, bs,
                                  sampler=training, num_workers=n_jobs)),
        Phase('valid', DataLoader(dataset, bs,
                                  sampler=valid, num_workers=n_jobs), grad=False),
        Phase('test', DataLoader(dataset, disp_batch,
                                 sampler=test, num_workers=n_jobs), grad=False),
    ]


def main():
    """
    """
    exp_path = Path.cwd()
    torch.cuda.empty_cache()
    # get user to choose training config
    config_file = "ResNet.json"
    # Load Test Parameters
    with open(config_file, "r") as f:
        x = json.load(f)
    hyperparams = x["HYPERPARAMETERS"]
    name = x["model"]

    n_bands = hyperparams['n_bands']
    patch_size = hyperparams['patch_size']
    _, path = utils.experiment_path_build(x)

    c_drive_docs = Path(x["LOGGING"]["log_location"])
    log_file = Path(x["LOGGING"]["log_file"])
    default_device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    # Parameters
    validation_split = hyperparams["validation_split"]
    test_split = hyperparams["test_split"]
    random_seed = x["BASIC_PARAMETERS"]["random_seed"]
    shuffle_dataset = x["LOCATIONS"]["shuffle_dataset"]
    disp_batch = hyperparams["disp_batch"]  # images to display on test
    epochs = hyperparams["epoch"]
    bs = hyperparams["batch_size"]
    n_jobs = x["BASIC_PARAMETERS"]["n_jobs"]
    lr = hyperparams["learning_rate"]

    HSI_ds = HSI_Dataset(path)
    _, classes = utils.get_classes(path)
    num_classes = len(classes)

    train_ds_sampler, valid_ds_sampler, test_ds_sampler = train_test_valid_split(
        HSI_ds, shuffle_dataset, validation_split, test_split, random_seed)

    phases = make_phases(HSI_ds, train_ds_sampler,
                         valid_ds_sampler, test_ds_sampler, bs=32, n_jobs=4, disp_batch=disp_batch)
    model = Net(ResidualBlock, [2, 4, 8], in_channels=n_bands,
                num_classes=num_classes).to(default_device)
    # Lee Model Call
    # model = Net(n_bands, n_classes, patch_size)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    training_params = {'Dataset Path': path,
                       'Experiment Path': exp_path,
                       'number of classes': num_classes,
                       'number of bands': n_bands,
                       'patch size': patch_size,
                       'test_train Split': {
                           'validation split': validation_split,
                           'shuffle dataset': shuffle_dataset,
                           'random seed': random_seed},
                       'Batch Size': bs,
                       "Number of Jobs": n_jobs,
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
        CSVLogger(c_drive_docs.joinpath(log_file), training_params),
        ProgressBar(),
        save_model(c_drive_docs, name),
        test_images(path=c_drive_docs, batch_size=5, classes=classes)
    ])
    # save_model(exp_path, name)
    train(model, opt, phases, cb, epochs=epochs,
          device=default_device, loss_fn=F.cross_entropy)
    lr_history = pd.DataFrame(cb['scheduler'].parameter_history('lr'))
    ax = lr_history.plot(figsize=(8, 6))
    ax.set_xlabel('Training Batch Index')
    ax.set_ylabel('Learning Rate')
    fig = ax.get_figure()
    file_loc = [str(c_drive_docs) + "\\checkpoints\\lr-test.jpg"]
    s = ""
    s = s.join(file_loc)
    conf_path = Path(s)
    fig.savefig(conf_path)


if __name__ == "__main__":
    main()
