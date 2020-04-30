import os
import torch
from abc import abstractmethod
import tempfile

import numpy as np
from torchvision import transforms

from utils.util import DeNormalize, lib_path, import_given_path
from utils.verification import evaluate_accuracy
from utils.logging_config import logger


class BaseMetric(torch.nn.Module):
    def __init__(self, output_key, target_key, nickname, scenario='training'):
        super().__init__()
        self.nickname = nickname
        self.output_key = output_key
        self.target_key = target_key
        self.scenario = scenario

    @abstractmethod
    def clear(self):
        """ Initialize variables needed for metrics calculations.

        This function would be called in TrainingWorker._init_output()
        See the TopKAcc below for example.
        """
        pass

    @abstractmethod
    def update(self, data, output):
        """ Update metric values in each batch.

        This function would be called inside torch.no_grad() in WorkerTemplate._update_all_metrics()
        """
        pass

    @abstractmethod
    def finalize(self):
        """ Calculate the final metric values given the variables updated in each batch. """
        pass


class TestMetric(BaseMetric):
    def __init__(self, k, output_key, target_key, nickname=None, scenario='training'):
        nickname = f'top{self.k}_acc_{target_key}' if nickname is None else nickname
        super().__init__(output_key, target_key, nickname, scenario)
        self.k = k

    def clear(self):
        self.total_correct = 0
        self.total_number = 0

    def update(self, data, output):
        self.total_correct += 2
        self.total_number += 1
        return self.total_correct / self.total_number

    def finalize(self):
        return self.total_correct / self.total_number


class VerificationMetric(BaseMetric):
    def __init__(self, output_key, target_key,
                 nickname=None, num_of_folds=5, scenario='validation'):
        nickname = f"verificatoin_acc_{target_key}" if nickname is None else nickname
        super().__init__(output_key, target_key, nickname, scenario)
        self.num_of_folds = num_of_folds
        self.cos_values = []
        self.is_same_ground_truth = []

    def clear(self):
        self.cos_values = []
        self.is_same_ground_truth = []

    def update(self, data, output):
        self.cos_values.append(output[self.output_key].cpu().numpy())
        self.is_same_ground_truth.append(data[self.target_key].cpu().numpy())
        return None

    def finalize(self):
        self.cos_values = np.concatenate(self.cos_values, axis=None)
        self.is_same_ground_truth = np.concatenate(self.is_same_ground_truth, axis=None)
        accuracy, threshold, roc_tensor = self.evaluate_and_plot_roc(
            self.cos_values, self.is_same_ground_truth, self.num_of_folds
        )
        logger.info(f">>>> In verification metric, accuracy:{accuracy}, threshold: {threshold}")
        return accuracy

    def evaluate_and_plot_roc(self, coses, issame, nrof_folds=5):
        accuracy, best_thresholds, roc_curve_tensor = evaluate_accuracy(
            coses, issame, nrof_folds
        )
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


class TopKAcc(BaseMetric):
    def __init__(self, k, output_key, target_key, nickname=None):
        nickname = f'top{self.k}_acc_{target_key}' if nickname is None else nickname
        super().__init__(output_key, target_key, nickname)
        self.k = k

    def clear(self):
        self.total_correct = 0
        self.total_number = 0

    def update(self, data, output):
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
    Module calculating FID score by saving all images into temporary directories
    """
    fid_score = import_given_path("fid_score", os.path.join(lib_path, 'pytorch_fid/fid_score.py'))

    def __init__(self, output_key, target_key, unnorm_mean=(0.5,), unnorm_std=(0.5,), nickname="FID_InceptionV3"):
        super().__init__(output_key, target_key, nickname)
        self.from_tensor_to_pil = transforms.Compose([
            DeNormalize(unnorm_mean, unnorm_mean),
            transforms.ToPILImage()
        ])
        self.tmp_gt_dir = tempfile.TemporaryDirectory(prefix='gt_')
        self.tmp_out_dir = tempfile.TemporaryDirectory(prefix='out_')

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
        return self.fid_score.calculate_fid_given_paths(
            paths=[self.tmp_gt_dir.name, self.tmp_out_dir.name],
            batch_size=10, cuda=True, dims=2048)


class FIDScore(BaseMetric):
    """
    Abstract class of FID score calculator (store inception activation in memory)
    """
    fid_score = import_given_path("fid_score", os.path.join(lib_path, 'pytorch_fid/fid_score.py'))

    def __init__(self, output_key, target_key, unnorm_mean=(0.5,), unnorm_std=(0.5,), nickname="FID_InceptionV3"):
        super().__init__(output_key, target_key, nickname)
        self._deNormalizer = DeNormalize(unnorm_mean, unnorm_mean)
        self._gt_activations = []
        self._out_activations = []

    def clear(self):
        self._gt_activations = []
        self._out_activations = []

    def _preprocess_tensor(self, tensor):
        tensor = self._deNormalizer(tensor)  # domain: [-1, 1] -> [0, 1]
        tensor = tensor.repeat(1, 3, 1, 1)   # convert 1-channel images to 3-channels
        return tensor

    @abstractmethod
    def _get_activation(self, tensors):
        pass

    def update(self, data, output):
        gt_tensors = self._preprocess_tensor(data[self.target_key])
        out_tensors = self._preprocess_tensor(output[self.output_key])
        self._gt_activations.append(self._get_activation(gt_tensors))
        self._out_activations.append(self._get_activation(out_tensors))
        return None

    def finalize(self):
        gt_activations = np.concatenate(self._gt_activations)
        out_activations = np.concatenate(self._out_activations)
        score = self._get_fid_score(gt_activations, out_activations)
        return score

    def _get_fid_score(self, gt_activations, out_activations):
        """
        Given two distribution of features, compute the FID score between them
        """
        m1 = np.mean(gt_activations, axis=0)
        m2 = np.mean(out_activations, axis=0)
        s1 = np.cov(gt_activations, rowvar=False)
        s2 = np.cov(out_activations, rowvar=False)
        return self.fid_score.calculate_frechet_distance(m1, s1, m2, s2)


class FIDScoreInceptionV3(FIDScore):
    inception = import_given_path("inception", os.path.join(lib_path, 'pytorch_fid/inception.py'))

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        block_idx = self.inception.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self._backbone = self.inception.InceptionV3([block_idx])
        self._backbone.eval()

    def _get_activation(self, tensors):
        return self._backbone(tensors)[0].squeeze().cpu().numpy()
