from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F

from model.xcos_modules import l2normalize


class GradientExtractor:
    """ Extracting activations and
    registering gradients from targetted intermediate layers
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, img1, img2):
        self.gradients = []
        feat1, out1 = self.forward(img1)
        feat2, out2 = self.forward(img2)
        return feat1, feat2, out1, out2

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward(self, x):
        feats = []
        x = self.model.input_layer(x)
        x = self.model.body(x)
        x.register_hook(self.save_gradient)
        feats.append(x)
        x = self.model.output_layer(x)
        return feats, x


class ModelOutputs:
    """ Making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers.
    """

    def __init__(self, model):
        self.model = model
        self.extractor = GradientExtractor(self.model)

    def get_grads(self):
        return self.extractor.gradients

    def __call__(self, img1, img2):
        feat1, feat2, out1, out2 = self.extractor(img1, img2)

        out1 = l2normalize(out1)
        out2 = l2normalize(out2)
        cos = F.cosine_similarity(out1, out2, dim=1, eps=1e-6)

        return feat1, feat2, cos


class FaceGradCam:

    def __init__(self, model):
        self.model = model
        self.extractor = ModelOutputs(self.model)

    def __call__(self, img1, img2):
        feat1, feat2, output = self.extractor(img1, img2)

        self.model.zero_grad()
        output.backward(retain_graph=True)
        grads = self.extractor.get_grads()
        hm1 = self.make_heatmap(grads[0].cpu().data.numpy(), feat1[0].cpu().data.numpy())
        hm2 = self.make_heatmap(grads[1].cpu().data.numpy(), feat2[0].cpu().data.numpy())

        return hm1, hm2

    def make_heatmap(self, grad, feat):
        """Batch operation supported
        """
        weights = np.mean(grad, axis=(-2, -1), keepdims=True)
        x = weights * feat

        x = x.sum(axis=1)
        x = np.maximum(0, x)
        x = x - np.min(x, axis=(-2, -1), keepdims=True)
        x = x / np.max(x, axis=(-2, -1), keepdims=True)
        x = 1. - x
        return x

    def make_img(self, heatmap, size, ori_img=None):
        """Batch operation NOT suppored
        """
        hm = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        hm = cv2.resize(hm, size)
        if ori_img is not None:
            hm = np.float32(hm) / 255 + np.transpose(ori_img.numpy(), (1, 2, 0)) * 0.5 + 0.5
            hm /= np.max(hm)
            hm = np.uint8(255 * hm)
        return Image.fromarray(hm)
