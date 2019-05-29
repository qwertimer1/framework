from Callback import Callback
import sys
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
            