
"""
loss module
Implements rolling loss as a callback
"""


from Callback import Callback


class RollingLoss(Callback):
    """Rolling loss callback to smooth out loss over time

    Arguments:
        Callback: super class Callback
    """

    def __init__(self, smooth=0.98):
        self.smooth = smooth

    def batch_ended(self, phase, **kwargs):
        """Implements loss smoothing at batch end.

        Arguments:
            phase: current phase of training
        """
        prev = phase.rolling_loss
        a = self.smooth
        avg_loss = a * prev + (1 - a) * phase.batch_loss
        debias_loss = avg_loss / (1 - a ** phase.batch_index)
        phase.rolling_loss = avg_loss
        phase.update(debias_loss)

    def epoch_ended(self, phases, **kwargs):
        """Updates loss metric for all phases

        Arguments:
            phases: Updates for all phases
        """
        for phase in phases:
            phase.update_metric('loss', phase.last_loss)
