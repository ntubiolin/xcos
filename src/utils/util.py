import os
import os.path as op
from glob import glob

import torch

lib_path = op.abspath(op.join(__file__, op.pardir, op.pardir, op.pardir, 'libs'))


def get_instance(module, name, config, *args, **kargs):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'], **kargs)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_everything_under(root_dir, pattern='*', only_dirs=False, only_files=False):
    assert not(only_dirs and only_files), 'You will get nothnig '\
        'when "only_dirs" and "only_files" are both set to True'
    everything = sorted(glob(os.path.join(root_dir, pattern)))
    if only_dirs:
        everything = list(filter(lambda f: os.path.isdir(f), everything))
    if only_files:
        everything = list(filter(lambda f: os.path.isfile(f), everything))
    return everything


def one_hot_embedding(labels, num_classes):
    # From https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/26
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
