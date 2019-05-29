from Callback import Callback
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict, OrderedDict

class ProgressBar(Callback):

    def training_started(self, phases, **kwargs):
        bars = OrderedDict()
        for phase in phases:
            bars[phase.name] = tqdm(total=len(phase.loader), desc=phase.name)
        self.bars = bars

    def batch_ended(self, phase, **kwargs):
        bar = self.bars[phase.name]
        bar.set_postfix_str(f'loss: {phase.last_loss:.4f}')
        bar.update(1)
        bar.refresh()

    def epoch_ended(self, **kwargs):
        for bar in self.bars.values():
            bar.n = 0
            bar.refresh()

    def training_ended(self, **kwargs):
        for bar in self.bars.values():
            bar.n = bar.total
            bar.refresh()
            bar.close()