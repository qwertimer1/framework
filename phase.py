from collections import OrderedDict


class Phase:
    """
    Model training loop phase.

    Each model's training loop iteration could be separated into (at least) two
    phases: training and validation. The instances of this class track
    metrics and counters, related to the specific phase, and keep the reference
    to subset of data, used during phase.
    """

    def __init__(self, name, loader, grad=True):
        self.name = name
        self.loader = loader
        self.grad = grad
        self.batch_loss = None
        self.batch_index = 0
        self.rolling_loss = 0
        self.losses = []
        self.metrics = OrderedDict()

    @property
    def last_loss(self):
        """
        Gets last loss value

        Returns:
            loss: returns loss result
        """
        return self.losses[-1] if self.losses else None

    @property
    def last_metrics(self):
        """
        gets last metrics

        Returns:
            metrics:
        """
        metrics = OrderedDict()
        metrics[f'{self.name}_loss'] = self.last_loss
        for name, values in self.metrics.items():
            metrics[f'{self.name}_{name}'] = values[-1]
        return metrics

    @property
    def metrics_history(self):
        """
        Stores the metrics history

        Returns:
            metrics:
        """
        metrics = OrderedDict()
        for name, values in self.metrics.items():
            metrics[f'{self.name}_{name}'] = values
        return metrics

    def update(self, loss):
        self.losses.append(loss)

    def update_metric(self, name, value):
        """
        Updates metrics related to the phase
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
