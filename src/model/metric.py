import os
import torch
from abc import abstractmethod
import importlib.util
import tempfile

from torchvision import transforms

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


class FIDScoreOffline(BaseMetric):
    """
    Module calculating FID score
    """

    def __init__(self, output_key, target_key, unnorm_mean=(0.5,), unnorm_std=(0.5,), nickname="FID_InceptionV3"):
        super().__init__(output_key, target_key, nickname)
        self.from_tensor_to_pil = transforms.Compose([
            UnNormalize(unnorm_mean, unnorm_mean),
            transforms.ToPILImage()
        ])
        self.tmp_gt_dir = tempfile.TemporaryDirectory(prefix='gt_')
        self.tmp_out_dir = tempfile.TemporaryDirectory(prefix='out_')

        # Load module fid_score given path
        spec = importlib.util.spec_from_file_location("fid_score", os.path.join(lib_path, 'pytorch_fid/fid_score.py'))
        self.fid_score = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fid_score)

    def clear(self):
        self.tmp_gt_dir.cleanup()
        self.tmp_out_dir.cleanup()
        self.tmp_gt_dir = tempfile.TemporaryDirectory(prefix='gt_')
        self.tmp_out_dir = tempfile.TemporaryDirectory(prefix='out_')

    def _save_img_tensor(self, tensor, buffer_dir):
        """ Save image tensor to a named temporary file and return the name."""
        temp_f = tempfile.NamedTemporaryFile(suffix='.png', dir=buffer_dir.name, delete=False)
        pil_image = self.from_tensor_to_pil(tensor.cpu())
        pil_image.save(temp_f)
        temp_f.close()

    def update(self, data, output):
        for gt_tensor, out_tensor in zip(data[self.target_key], output[self.output_key]):
            self._save_img_tensor(gt_tensor, self.tmp_gt_dir)
            self._save_img_tensor(out_tensor.clamp(-1, 1), self.tmp_out_dir)
        return None

    def finalize(self):
        fid_score = self.fid_score.calculate_fid_given_paths(
            paths=[self.tmp_gt_dir.name, self.tmp_out_dir.name],
            batch_size=10, cuda=True, dims=2048)
        return fid_score
