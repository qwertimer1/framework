"""
training module for training the neural network
"""
import torch
from torch.nn import functional as F


def train(model, opt, phases, callbacks=None, epochs=1, device=False, loss_fn=F.nll_loss):
    """    A generic structure of training loop.

    Arguments:
        model {[type]} -- [description]
        opt {[type]} -- [description]
        phases {[type]} -- [description]

    Keyword Arguments:
        callbacks {[type]} -- [description] (default: {None})
        epochs {int} -- [description] (default: {1})
        device {bool} -- [description] (default: {False})
        loss_fn {[type]} -- [description] (default: {F.nll_loss})
    """

    model.to(device)

    cb = callbacks

    cb.training_started(phases=phases, optimizer=opt)

    for epoch in range(1, epochs + 1):
        cb.epoch_started(epoch=epoch)

        for phase in phases:
            # If phase not test
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
                       optimizer=opt, model=model, loss=loss, output=out, target=y)

    cb.training_ended(phases=phases, model=model, target=y)


def place_and_unwrap(batch, dev):
    """Unpacks data and target from batch and rearranges the data into ()

    Arguments:
        batch: current batch of data
        dev: Device (default: cuda)

    Returns:
        x: unpacked input data
        y: target data as tensor
    """

    x, *y = batch
    #x = x.permute(0, 3, 1, 2)  # reshape f
    x = x.to(dev)

    y = [tensor.to(dev, dtype=torch.int64) for tensor in y]
    if len(y) == 1:
        [y] = y
    return x, y
