import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, output_key, target_key, nickname=None, weight=1):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_key = output_key
        self.target_key = target_key
        self.weight = weight
        self.nickname = self.__class__.__name__ if nickname is None else nickname

    def forward(self, data, output):
        logits = output[self.output_key]
        target = data[self.target_key]
        return self.loss_fn(logits, target)
