import os
from glob import glob

import torch


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
