import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

import torch.nn as nn
import torch.nn.functional as F

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
