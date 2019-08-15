import os
import torch
from abc import abstractmethod
import importlib.util

from utils.util import UnNormalize, lib_path


class BaseMetric(torch.nn.Module):
    def __init__(self, output_key, target_key, nickname):
        super().__init__()
        self.nickname = nickname
        self.output_key = output_key
        self.target_key = target_key

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def update(self, data, output):
        pass

    @abstractmethod
    def finalize(self):
        pass


class TopKAcc(BaseMetric):
    def __init__(self, k, output_key, target_key, nickname=None):
        nickname = f'top{self.k}_acc_{target_key}' if nickname is None else nickname
        super().__init__(output_key, target_key, nickname)
        self.k = k

    def clear(self):
        self.total_correct = 0
        self.total_number = 0

    def update(self, data, output):
        with torch.no_grad():
            logits = output[self.output_key]
            target = data[self.target_key]
            pred = torch.topk(logits, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item()
        self.total_correct += correct
        self.total_number += len(target)
        return correct / len(target)

    def finalize(self):
        return self.total_correct / self.total_number


class FIDScore(torch.nn.Module):
    spec = importlib.util.spec_from_file_location("fid_score", os.path.join(lib_path, 'pytorch_fid/fid_score.py'))
    fid_score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fid_score)

    def __init__(self, output_key, target_key, unnorm_mean=(0.5,), unnorm_std=(0.5,), nickname="FID_InceptionV3"):
        super().__init__()
        self.nickname = nickname
        self.output_key = output_key
        self.target_key = target_key
        self.unnormalizer = UnNormalize(unnorm_mean, unnorm_mean)

    def clear(self):
        # TODO: create containers of temporary files using TemporaryDirectory, etc. in package tempfile
        pass

    def update(self, data, output):
        # TODO: unnormalize real and fake images, save them to temporary files.
        return None

    def finalize(self):
        # TODO: calculate fid scores using functions in libs/pytorch_fid/fid_score.py
        return 0
