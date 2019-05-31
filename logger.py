from Callback import Callback
import sys
import os
import io
import six
import numpy as np
import csv

from collections import defaultdict, OrderedDict


class StreamLogger(Callback):
    """
    Writes performance metrics collected during the training process into list
    of streams.

    Parameters:
        streams: A list of file-like objects with `write()` method.

    """

    def __init__(self, streams=None, log_every=1):
        self.streams = streams or [sys.stdout]
        self.log_every = log_every

    def epoch_ended(self, phases, epoch, **kwargs):
        metrics = merge_dicts([phase.last_metrics for phase in phases])
        values = [f'{k}={v:.4f}' for k, v in metrics.items()]
        values_string = ', '.join(values)
        string = f'Epoch: {epoch:4d} | {values_string}\n'
        for stream in self.streams:
            stream.write(string)
            stream.flush()


def merge_dicts(ds):
    merged = OrderedDict()
    for d in ds:
        for k, v in d.items():
            merged[k] = v
    return merged


class CSVLogger(Callback):
    """
    Writes performance metrics collected during the training process into list
    of streams.

    Parameters:
        streams: A list of file-like objects with `write()` method.

    """

    def __init__(self, filename="", training_params="", streams=None, log_every=1):
        self.log_every = log_every
        self.filename = filename
        self.training_params = training_params

    def training_started(self, phases,  ** kwargs):
        """
        Print hyperparameters and other information

        """
        print(self.training_params)
        with open(self.filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.training_params])

    def epoch_ended(self, phases, epoch, **kwargs):
        metrics = merge_dicts([phase.last_metrics for phase in phases])
        values = [f'{k}={v:.4f}' for k, v in metrics.items()]
        values_string = ', '.join(values)
        string = f'Epoch: {epoch:4d} | {values_string}\n'
        with open(self.filename, 'a') as csvfile:
            print(csvfile)
            writer = csv.writer(csvfile)
            writer.writerow([string])
