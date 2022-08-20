from typing import List
import torch
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', avg_as_val=False):
        self.name = name
        self.fmt = fmt
        self.avg_as_val = avg_as_val  # dirty trick to make my script work (I'm so tired help)
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.avg_as_val:
            self.avg = val
        else:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class TensorboardMeter(object):
    def __init__(self, experiment: str):
        self.writer = SummaryWriter(experiment)

        # temporary variable that stores all train values
        self.tmp_trainvals = {}

    def update_train(self, meters: List[AverageMeter]):
        """Stores all values from the AverageMeters of the training step.

        Args:
            meters (List[AverageMeter]): AverageMeters from the training step
        """
        for meter in meters:
            self.tmp_trainvals[meter.name] = meter.avg

    def update_val(self, meters: List[AverageMeter], epoch: int):
        """Saves all train-validation pairs in the Tensorboard logdir

        Args:
            meters (List[AverageMeter]): AverageMeters from the validation step
            epoch (int): epoch number to write in the Tensorboard logdir
        """
        for meter in meters:
            # Only writes the AverageMeters that were already present in the training step
            if meter.name in self.tmp_trainvals:
                self.writer.add_scalars(
                    meter.name,
                    {
                        'train': self.tmp_trainvals[meter.name],
                        'val': meter.avg
                    },
                    epoch
                )

        # Ensures the writer completes the validation step
        self.writer.flush()

    def close(self):
        self.writer.close()
