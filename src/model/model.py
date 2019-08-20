import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .networks import MnistGenerator, MnistDiscriminator


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


class MnistGAN(BaseModel):
    def __init__(self, spectral_normalization=True, d=128):
        super().__init__()
        self.generator = MnistGenerator(d=d)
        self.discriminator = MnistDiscriminator(spectral_normalization=spectral_normalization, d=d)

        self.generator.weight_init(mean=0.0, std=0.02)
        self.discriminator.weight_init(mean=0.0, std=0.02)

    def forward(self, data_dict, scenario):
        x = data_dict['data_input']
        batch_size = x.size(0)

        # Generate images from random vector z. When inferencing, it's the only thing we need.
        z = torch.randn((batch_size, 100)).view(-1, 100, 1, 1).to(x.device)
        G_z = self.generator(z)
        model_output = {"G_z": G_z}
        if scenario == 'generator_only':
            return model_output

        # Feed fake images to the discriminator. When training generator, it's the last thing we need.
        D_G_z = self.discriminator(G_z).squeeze()
        model_output["D_G_z"] = D_G_z
        if scenario == 'generator':
            return model_output

        # Feed real images the discriminator. Only when training discriminator will this be needed.
        assert scenario == 'discriminator'
        D_x = self.discriminator(x).squeeze()
        model_output["D_x"] = D_x
        return model_output
