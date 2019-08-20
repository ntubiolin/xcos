import os
import os.path as op
from glob import glob
import importlib.util

import torch
import numpy as np

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


class InverseNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image(s) to be normalized.
            Should be in size [B, C, W, H] (a batch of images) or [C, W, H] (single image)
        Returns:
            Tensor: Normalized image.
        """

        if len(tensor.shape) == 4:  # [B, C, W, H]
            c_dim = 1
        elif len(tensor.shape) == 3:  # [C, W, H]
            c_dim = 0
        else:
            raise NotImplementedError()

        tensors = tensor.split(1, dim=c_dim)
        out = []
        for t, m, s in zip(tensors, self.mean, self.std):
            # Normalization: (t - m) / s
            out.append(t * s + m)
        tensor = torch.cat(out, dim=c_dim)
        return tensor


def import_given_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_np_histogram(tensor):
    return np.histogram(tensor.cpu().numpy().flatten())
