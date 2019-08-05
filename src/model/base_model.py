import torch.nn as nn
import numpy as np

from utils.logging_config import logger


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info(f'Trainable parameters: {params}')
        logger.info(self)
