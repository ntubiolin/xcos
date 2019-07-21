import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, output_key, target_key, nickname=None, weight=1):
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.weight = weight
        self.nickname = self.__class__.__name__ if nickname is None else nickname

    def _preproces(self, data_dict, output_dict):
        return data_dict, output_dict

    def _postprocess(self, output, target):
        return output, target

    def forward(self, data_dict, output_dict):
        data_dict, output_dict = self._preproces(data_dict, output_dict)
        output = output_dict[self.output_key]
        target = data_dict[self.target_key]
        output, target = self._postprocess(output, target)
        return self.loss_fn(output, target)


class CrossEntropyLoss(BaseLoss):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.loss_fn = nn.CrossEntropyLoss()
