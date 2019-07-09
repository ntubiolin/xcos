import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, output_key='verb_logits', target_key='verb_class'):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, data, output):
        logits = output[self.output_key]
        target = data[self.target_key]
        return self.loss_fn(logits, target)


class FocalLoss(nn.Module):
    def __init__(self, output_key='verb_logits', target_key='verb_class', gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.gamma = gamma
        self.nll_loss = torch.nn.NLLLoss(weight)

    def forward(self, data, output):
        logits = output[self.output_key]
        target = data[self.target_key]

        return self.nll_loss((1 - F.softmax(logits, 1)) ** self.gamma * F.log_softmax(logits, 1), target)
