import json

from pathlib import Path


import pandas as pd
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader


from Accuracy import Accuracy
from Callback import Callback, CallbacksGroup
from datasets import Whale_Image_Dataset
from logger import StreamLogger, CSVLogger
from ResNet import ResNet, ResidualBlock
#from net import ResNet as Net
from saving import save_model
import Parameters_schedule
from Parameters_schedule import Scheduler, OneCycleSchedule
from phase import Phase
from train import train
from utils import ProgressBar
from utils import utils#, train_test_valid_split
from loss import RollingLoss
from databunch import get_data


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

def make_phases_databunch(dl, bs=32, n_jobs=0, disp_batch=5):
    """
    make phases builds the dataloaders for each training phase and implements flags for some callback methods.
    """
    return [
        Phase('train', dl.train_dl),
        Phase('valid', dl.valid_dl, grad=False),
        Phase('test', dl.valid_dl, grad=False),
    ]


def main():
    torch.cuda.empty_cache()
    default_device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    exp_path = Path.cwd()
    torch.cuda.empty_cache()
    # get user to choose training config
    config_file = "training_config.json"

    # Load Test Parameters
    with open(config_file, "r") as f:
        x = json.load(f)

    hyperparams = x["HYPERPARAMETERS"]
    name = x["model"]
    exp_loc = Path(x["LOGGING"]["log_location"])

    n_classes = hyperparams['n_classes']
    patch_size = hyperparams['patch_size']


    _, path = utils.experiment_path_build(x)
    log_file = Path(x["LOGGING"]["log_file"])


    # Parameters
    validation_split = hyperparams["validation_split"]
    n_bands = x["BASIC_PARAMETERS"]["n_bands"]
    test_split = hyperparams["test_split"]
    random_seed = x["BASIC_PARAMETERS"]["random_seed"]
    shuffle_dataset = x["LOCATIONS"]["shuffle_dataset"]
    disp_batch = hyperparams["disp_batch"]  # images to display on test
    epochs = hyperparams["epoch"]
    bs = hyperparams["batch_size"]
    n_jobs = x["BASIC_PARAMETERS"]["n_jobs"]
    lr = hyperparams["learning_rate"]


    csv_file = str(path) +'/' + 'Whalemanifest.csv' 
    
    ds = Whale_Image_Dataset(path, csv_file)
    # Params need moving
    #Hard code for now. Need to implement path to allow the user to choose the data
    data = get_data()
    print(data.c_in)
  
    #train_ds_sampler, valid_ds_sampler, test_ds_sampler = train_test_valid_split(
    #    ds, shuffle_dataset, validation_split, test_split, random_seed)
 

#    phases = make_phases(ds, train_ds_sampler,
#                        valid_ds_sampler, test_ds_sampler, bs=bs, n_jobs=n_jobs)
    phases = make_phases_databunch(data, bs=bs, n_jobs=n_jobs)  
   

    model = ResNet(block = ResidualBlock, layers = [2,4,8], in_channels=n_bands, num_classes=n_classes)
    opt = optim.Adam(model.parameters(), lr=lr)
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
        CSVLogger(exp_loc.joinpath(log_file), training_params),
        ProgressBar(),
        save_model(exp_loc, name)
    ])

    train(model, opt, phases, cb, epochs=epochs,
          device=default_device, loss_fn=F.cross_entropy)
    lr_history = pd.DataFrame(cb['scheduler'].parameter_history('lr'))
    ax = lr_history.plot(figsize=(8, 6))
    ax.set_xlabel('Training Batch Index')
    ax.set_ylabel('Learning Rate')
    fig = ax.get_figure()
    file_loc = [str(exp_loc) + "\\checkpoints\\lr-test.jpg"]
    s = ""
    s = s.join(file_loc)
    conf_path = Path(s)
    fig.savefig(conf_path)


if __name__ == "__main__":
    main()
