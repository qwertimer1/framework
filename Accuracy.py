"""
accuracy creates a callack to provide a continuous Accuracy Evaluation
"""
from collections import defaultdict

from Callback import Callback


def accuracy(out, y_true):
    """
    Accuracy function calculates the accuracy of the model output against the true y values
    args: 
        out: model output
        y_true: true y values

    returns:
        match.float().mean()
    """
    y_hat = out.argmax(dim=-1).view(y_true.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)
    match = y_hat == y_true
    return match.float().mean()


class Accuracy(Callback):
    """
    Accuracy Callback class builds a callback class for calculating the accuracy of the model.
    Args:
        None
    """

    def epoch_started(self, **kwargs):
        """
        instatiate values and counts on epoch start
        """
        self.values = defaultdict(int)
        self.counts = defaultdict(int)

    def batch_ended(self, phase, output, target, **kwargs):
        """
        At batch end calculate accuracy
        Args:
            phase: training phase(training, valid, test)
            output: output of model
            target: ground truth

        """
        acc = accuracy(output, target).detach().item()
        self.counts[phase.name] += target.size(0)
        self.values[phase.name] += target.size(0) * acc

    def epoch_ended(self, phases, **kwargs):
        """
        At epoch end update metrics and send to phase module
        Args:
            phases: training phase(training, valid, test)
        """
        for phase in phases:
            metric = self.values[phase.name] / self.counts[phase.name]
            phase.update_metric('accuracy', metric)
