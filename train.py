
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
from Callback import Callback


def train(model, opt, phases, callbacks=None, epochs=1, device=False, loss_fn=F.nll_loss):
    """
    A generic structure of training loop.
    """
    model.to(device)

    cb = callbacks

    cb.training_started(phases=phases, optimizer=opt)

    for epoch in range(1, epochs + 1):
        cb.epoch_started(epoch=epoch)

        for phase in phases:
            n = len(phase.loader)
            cb.phase_started(phase=phase, total_batches=n)
            is_training = phase.grad
            model.train(is_training)

            for batch in phase.loader:

                phase.batch_index += 1
                cb.batch_started(phase=phase, total_batches=n)
                x, y = place_and_unwrap(batch, device)

                with torch.set_grad_enabled(is_training):
                    cb.before_forward_pass()
                    out = model(x)
                    cb.after_forward_pass()
                    loss = loss_fn(out, y)

                if is_training:
                    opt.zero_grad()
                    cb.before_backward_pass()
                    loss.backward()
                    cb.after_backward_pass()
                    opt.step()

                phase.batch_loss = loss.item()
                cb.batch_ended(phase=phase, output=out, target=y)

            cb.phase_ended(phase=phase)

        cb.epoch_ended(phases=phases, epoch=epoch,
                       optimizer=opt, model=model, loss=loss)

    cb.training_ended(phases=phases, model=model)


def place_and_unwrap(batch, dev):
    x, *y = batch
    x = x.permute(0, 3, 1, 2)  # reshape f
    x = x.to(dev)
    y = [tensor.to(dev) for tensor in y]
    if len(y) == 1:
        [y] = y
    return x, y
