import os
from glob import glob

import torch
import numpy as np


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


def replace_module_prefix(state_dict, prefix='module.', replace=''):
    new_state = {}
    for key in state_dict.keys():
        if key.startswith(prefix):
            new_key = replace + key[len(prefix):]
        else:
            new_key = key
        new_state[new_key] = state_dict[key]
    return new_state


def flow_to_image(flows):
    # https://github.com/vt-vl-lab/pytorch_flownet2/blob/master/FlowNet2_src/flowlib.py
    UNKNOWN_FLOW_THRESH = 1e7
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    def make_color_wheel():
        """
        Generate color wheel according Middlebury color code
        :return: Color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR

        colorwheel = np.zeros([ncols, 3])

        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
        col += RY

        # YG
        colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
        colorwheel[col:col + YG, 1] = 255
        col += YG

        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
        col += GC

        # CB
        colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
        colorwheel[col:col + CB, 2] = 255
        col += CB

        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
        col += + BM

        # MR
        colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col + MR, 0] = 255

        return colorwheel

    def compute_color(u, v):
        """
        compute optical flow color map
        :param u: optical flow horizontal map
        :param v: optical flow vertical map
        :return: optical flow in color code
        """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u ** 2 + v ** 2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

        return img
    new_flow_imgs = np.zeros([flows.shape[0], 3, flows.shape[2], flows.shape[3]])

    for iidx in range(flows.shape[0]):
        # u = flows[iidx * 2].cpu().numpy()
        # u = u.transpose([1, 2, 0])
        # u = np.dot(u[..., :3], [0.299, 0.587, 0.114])
        # v = flows[iidx * 2 + 1].cpu().numpy()
        # v = v.transpose([1, 2, 0])
        # v = np.dot(v[..., :3], [0.299, 0.587, 0.114])
        u = flows[iidx][0].cpu().numpy()
        v = flows[iidx][1].cpu().numpy()

        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.

        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))

        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))

        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)

        img = compute_color(u, v)

        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0
        img = img.transpose((2, 0, 1))

        new_flow_imgs[iidx] = np.uint8(img)

    return torch.from_numpy(new_flow_imgs)


def extract_missing_and_unexpected_keys(source_keys, target_keys):
    unexpected = [key for key in source_keys if key not in target_keys]
    missing = [key for key in target_keys if key not in source_keys]
    return missing, unexpected


# Channel when transformed
channel_nb = {'flow': 1, 'rgb': 3, 'bbox': 1, 'handpose': 3}
# Final channels
in_channel_map = {'rgb': 3, 'flow': 2, 'bbox': 1, 'handpose': 3}


def count_in_channels(modalities):
    in_channels = 0
    for modality in modalities:
        in_channels += in_channel_map[modality]
    return in_channels


def softmax(data):
    data = torch.tensor(np.array(data))
    data = torch.softmax(data, dim=1).numpy()
    assert np.isclose(data[0].sum(), 1)
    assert np.isclose(data[-1].sum(), 1)
    return data
