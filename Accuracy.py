from Callback import Callback
from collections import defaultdict

def accuracy(out, y_true):
    y_hat = out.argmax(dim=-1).view(y_true.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)
    match = y_hat == y_true
    return match.float().mean()

 
class Accuracy(Callback):

    def epoch_started(self, **kwargs):
        self.values = defaultdict(int)
        self.counts = defaultdict(int)

    def batch_ended(self, phase, output, target, **kwargs):
        acc = accuracy(output, target).detach().item()
        self.counts[phase.name] += target.size(0)
        self.values[phase.name] += target.size(0) * acc

    def epoch_ended(self, phases, **kwargs):
        for phase in phases:
            metric = self.values[phase.name] / self.counts[phase.name]
            phase.update_metric('accuracy', metric)   