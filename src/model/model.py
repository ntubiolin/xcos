import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model.base_model import BaseModel


class MnistModel(BaseModel):
    """
    Mnist model demo
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, data_dict):
        x = data_dict['data_input']
        c1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        c2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(c1)), 2))
        c2_flatten = c2.view(-1, 320)
        c2_activation = F.relu(self.fc1(c2_flatten))
        c2_dropout = F.dropout(c2_activation, training=self.training)
        fc_out = self.fc2(c2_dropout)
        out = F.log_softmax(fc_out, dim=1)
        return {
            "model_output": out
        }


class MnistGenerator(nn.Module):
    # architecture reference: https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_MNIST_DCGAN.py  # NOQA
    def __init__(self, d=128):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x


class MnistDiscriminator(nn.Module):
    # architecture reference: https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_MNIST_DCGAN.py  # NOQA
    def __init__(self, d=128, spectral_normalization=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

        if spectral_normalization:
            for attr_name in [f'conv{i}' for i in range(1, 6)]:
                new_attr = spectral_norm(getattr(self, attr_name))
                setattr(self, attr_name, new_attr)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))
        return x


class MnistGAN(BaseModel):
    def __init__(self, spectral_normalization=True):
        super().__init__()
        self.generator = MnistGenerator()
        self.discriminator = MnistDiscriminator(spectral_normalization=spectral_normalization)

    def forward(self, data_dict):
        x = data_dict['data_input']
        batch_size = x.size(0)
        z = torch.randn((batch_size, 100)).view(-1, 100, 1, 1).to(self.device)
        G_z = self.generator(z)
        D_G_z = self.discriminator(G_z)
        D_x = self.discriminator(x)

        model_output = {
            "G_z": G_z,
            "D_G_z": D_G_z,
            "D_x": D_x
        }
        return model_output

    @property
    def generator_module(self):
        return self.generator

    @property
    def discriminator_module(self):
        return self.discriminator
