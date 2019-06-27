"""
Module for saving the current checkpoints and the final model using the callback event handler
"""

import datetime as dt
from pathlib import Path
import torch

from Callback import Callback


class save_model(Callback):
    """save_model:
    Saves trained model
    """

    def __init__(self, training_path, model_save):
        """instatiates save locations

        Arguments:
                model_save: Experiment Name
                training_path: path of experiment logs

        """
        self.model_save = model_save
        self.training_path = training_path

    def epoch_ended(self, epoch, optimizer, model, loss, **kwargs):
        """Saves current network parameters at epoch end

        Arguments:
            epoch {[type]} -- current epoch
            optimizer {[type]} -- optimizer function
            model {[type]} -- network being trained
            loss {[type]} -- current loss
        """

        file_loc = [str(self.training_path), '\\checkpoints', '\\', str(
            self.model_save), '_', str(epoch), '.pth.tar']
        s = ""
        s = s.join(file_loc)
        PATH = Path(s)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,

        },
            PATH)
        print(f"Checkpoint Saved : {PATH}")

    def training_ended(self, model, **kwargs):
        """Saves model at training end

        Arguments:
            model: Model output
        """
        file_loc = [str(self.training_path), '\\checkpoints', '\\', str(
            self.model_save), '_', str(dt.datetime.now().strftime("%Y%m%d%H%M%S")), '.pt']  # NEED TO FIX
        s = ""
        s = s.join(file_loc)
        PATH = Path(s)
        final_model = Path(PATH)
        torch.save(model, final_model)
        print(f'Experiment Results saved {final_model}')
