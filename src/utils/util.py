import os
import os.path as op
from glob import glob
import importlib.util

import torch
import numpy as np

import io
import cv2
import base64
import seaborn as sns
from PIL import Image
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt


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


class DeNormalize(object):
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


def batch_visualize_xcos(img1s, img2s, grid_cos_maps, attention_maps):
    result_imgs = []
    for i in range(len(img1s)):
        result_imgs.append(visualize_xcos(img1s[i], img2s[i],
                                          grid_cos_maps[i], attention_maps[i]))
    return result_imgs


def visualize_xcos(image1, image2, grid_cos_map, attention_map,
                   name1=None, name2=None,
                   regressed_cos=None, is_same=None, threshold=0.245,
                   return_base64=False):
    """Plot the qualitative result of xCos

    Arguments:
        image1 [np.array] -- of shape (c, h, w); value: ![0, 255] (float32)
        image2 [np.array] -- of shape (c, h, w)
        grid_cos_map [np.array]  -- of shape (h, w)
        attention_map [np.array] -- of shape (h, w)

    Returns:
        [type] -- [description]
    """
    plt.gcf().clear()
    # name1, name2 = 'Left', 'Right'
    # isSame = int(isSame)

    # Unnormalize images
    image1 = ((image1 * 0.5 + 0.5) * 255).astype('uint8')
    image2 = ((image2 * 0.5 + 0.5) * 255).astype('uint8')
    # CHW2HWC
    image1 = np.transpose(image1, (1, 2, 0))
    image2 = np.transpose(image2, (1, 2, 0))
    # XXX BGR2RGB should be executed twice otherwise cv2 would complain in drawGird
    # Input img should be in PIL format (RGB).
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # same = 1 if float(cos_fr) > threshold else 0
    # title_str = getTFNPString(isSame, same)

    # Create visualization
    fig_size = (14, 3)
    fig, axs = plt.subplots(1, 4, tight_layout=True, figsize=fig_size)

    # if subtitle:
    #     fig.suptitle(title_str +
    #                     ' Cos=%.2f xCos=%.2f' % (float(cos_fr), cos_x))

    [axs[i].set_axis_off() for i in range(4)]
    axs[0].set_title('Face 1', y=-0.1)
    axs[1].set_title('Face 2', y=-0.1)
    axs[2].set_title(r'$cos_{patch}$', y=-0.1)
    axs[3].set_title(r'$weight_{attetion}$', y=-0.1)

    drawGridLines(image1, 6, 6)
    drawGridLines(image2, 6, 6)
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    # Show grid_cos_map.
    im, cbar = heatmap_seaborn(grid_cos_map, [], [], ax=axs[2],
                               cmap="RdBu", threshold=threshold)
    # Show weights_attention.
    im, cbar = heatmap(attention_map, [], [], ax=axs[3], cmap="YlGn")

    # Save file.
    # img_name = os.path.join(exPath, filename)
    # print(img_name)
    # score_log_name = os.path.splitext(img_name)[0]+'.txt'
    # with open(score_log_name, 'w') as the_file:
    #     the_file.write(f"{cos_x}")
    # plt.savefig(img_name, bbox_inches='tight')

    # return the base64 image (for demo xCos purpose)
    if return_base64:
        pic_IObytes = io.BytesIO()
        plt.savefig(pic_IObytes, format='jpg')
        pic_IObytes.seek(0)
        pic_hash = base64.b64encode(pic_IObytes.read())
        plt.close()
        return pic_hash
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', dpi=100)
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        plt.close()
        return image


def drawGridLines(image_t, w_lines=5, h_lines=6,
                  colorRGB=(128, 128, 128)):
    '''
    colorRGB: default: gray(128, 128, 128), you can use red(255, 0, 0)
    '''
    colorRGB = (255, 0, 0)
    w_lines += 1
    h_lines += 1
    h, w, _ = image_t.shape
    w_unit = int(w // w_lines)
    # w_start = int(w_unit // 2)
    w_start = w_unit
    h_unit = int(h // h_lines)
    # h_start = int(h_unit // 2)
    h_start = h_unit
    # Draw vertical grid lines
    for step in range(w_lines):
        start_pt = (w_start + w_unit * step, 0)
        end_pt = (w_start + w_unit * step, h)
        cv2.line(image_t, start_pt, end_pt, colorRGB, 1, 1)
    # Draw horizontal grid lines
    for step in range(h_lines):
        start_pt = (0, h_start + h_unit * step)
        end_pt = (w, h_start + h_unit * step)
        cv2.line(image_t, start_pt, end_pt, colorRGB, 1, 1)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if cbar_kw is None:
        cbar_kw = {}
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def heatmap_seaborn(data, row_labels, col_labels, ax=None,
                    cmap=None, cbarlabel="", threshold=0.5, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        exit('no ax')
        ax = plt.gca()

    # Plot the heatmap
    g = sns.heatmap(data, ax=ax, center=threshold, vmin=-1, vmax=1,
                    cmap=cmap, cbar_kws={'label': cbarlabel})

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar = None

    # # We want to show all ticks...
    # ax.set_xticks(np.arange(data.shape[1]))
    # ax.set_yticks(np.arange(data.shape[0]))
    # # ... and label them with the respective list entries.
    # ax.set_xticklabels(col_labels)
    # ax.set_yticklabels(row_labels)

    # # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return g, cbar
