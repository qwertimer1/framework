import torch
from Callback import Callback
import time
from pathlib import Path


class save_model(Callback):
    """[save_model]:

    Args:
    training_path: path of experiment logs
    model_save: Experiment Name
    """

    def __init__(self, training_path, model_save):
        self.model_save = model_save
        self.training_path = training_path

    def epoch_ended(self, epoch, optimizer, model, loss,  **kwargs):

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
        final_model = Path(self.model_save + str(time.time) + 'pt')
        torch.save(model, final_model)
        print(f'Experiment Results saved {final_model}')
