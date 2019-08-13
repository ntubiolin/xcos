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
    def __init__(self, spectral_normalization=True):
        super().__init__()
        self.generator = MnistGenerator()
        self.discriminator = MnistDiscriminator(spectral_normalization=spectral_normalization)

        self.generator.weight_init(mean=0.0, std=0.02)
        self.discriminator.weight_init(mean=0.0, std=0.02)

    def forward(self, data_dict, network_name):
        x = data_dict['data_input']
        batch_size = x.size(0)
        z = torch.randn((batch_size, 100)).view(-1, 100, 1, 1).to(x.device)
        G_z = self.generator(z)
        D_G_z = self.discriminator(G_z).squeeze()
        model_output = {
            "G_z": G_z,
            "D_G_z": D_G_z
        }
        if network_name == 'discriminator':
            D_x = self.discriminator(x).squeeze()
            model_output["D_x"] = D_x
        return model_output

    @property
    def network_names(self):
        return ['generator', 'discriminator']
        return self.discriminator
