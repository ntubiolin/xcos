import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from torchvision.utils import make_grid

from data_loader.base_data_loader import BaseDataLoader
from pipeline.base_pipeline import BasePipeline
from utils.global_config import global_config


class WorkerTemplate(ABC):
    """ Worker template, base class for trainer, validator and tester.

    Child class need to implement at least the _run_and_optimize_model() method
    that deals with the main optimization & model inference.
    """
    def __init__(
        self, pipeline: BasePipeline, data_loader: BaseDataLoader, step: int
    ):
        # Attributes listed below are shared from pipeline among all different workers.
        for attr_name in ['device', 'model', 'evaluation_metrics', 'writer']:
            setattr(self, attr_name, getattr(pipeline, attr_name))

        self.log_step = global_config['trainer']['log_step']
        self.verbosity = global_config['trainer']['verbosity']

        self.data_loader = data_loader
        self.step = step  # Tensorboard log step

    # ============ Implement the following functions ==============
    @property
    @abstractmethod
    def enable_grad(self):
        pass

    @abstractmethod
    def _run_and_optimize_model(self, data):
        """ Put data into model and optimize the model"""
        return {}, None, []

    def _print_log(self, epoch, batch_idx, batch_start_time, loss, metrics):
        """ Print messages on terminal. """
        pass

    @abstractmethod
    def _setup_model(self):
        """ Set random seed and self.model.eval() or self.model.train() """
        pass

    def _write_images(self, data, model_output):
        """ Write images to Tensorboard """
        self.writer.add_image("data_input", make_grid(data["data_input"], nrow=4, normalize=True))

    # ============ Implement the above functions ==============

    # Generally, the following function should not be changed.

    def _setup_writer(self):
        """ Setup Tensorboard writer for each iteration """
        self.writer.set_step(self.step, self.data_loader.name)
        self.step += 1

    def _get_and_write_loss(self, data, model_output):
        """ Calculate losses and write them to Tensorboard

        Losses will be summed and returned.
        """
        losses = []
        for loss_function in self.loss_functions:
            if loss_function.weight <= 0.0:
                continue
            loss = loss_function(data, model_output) * loss_function.weight
            losses.append(loss)
            self.writer.add_scalar(f'{loss_function.nickname}', loss.item())
        total_loss = sum(losses)
        self.writer.add_scalar('total_loss', total_loss.item())
        return total_loss

    def _get_and_write_metrics(self, data, model_output):
        """ Calculate evaluation metrics and write them to Tensorboard """
        acc_metrics = np.zeros(len(self.evaluation_metrics))
        for i, metric in enumerate(self.evaluation_metrics):
            acc_metrics[i] += metric(data, model_output)
            self.writer.add_scalar(metric.nickname, acc_metrics[i])
        return acc_metrics

    def _data_to_device(self, data):
        """ Put data into CPU/GPU """
        for key in data.keys():
            # Dataloader yeilds something that's not tensor, e.g data['video_id']
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(self.device)
        return data

    def _stats_init(self):
        """ Initialize epoch statistics like elapsed time, total loss, and metrics """
        epoch_start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.evaluation_metrics))
        return epoch_start_time, total_loss, total_metrics

    def _stats_update(self, stats, products):
        """ Update epoch statistics """
        loss, metrics = products['loss'], products['metrics']
        epoch_start_time, total_loss, total_metrics = stats
        total_loss += loss.item()
        total_metrics += metrics
        return epoch_start_time, total_loss, total_metrics

    def _stats_finalize(self, stats):
        """ Calculate the overall elapsed time and average loss/metrics in this epoch """
        epoch_start_time, total_loss, total_metrics = stats
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        return epoch_time, avg_loss, avg_metrics

    def _iter_data(self, epoch):
        """ Iterate through the dataset and do inference, calculate losses and metrics (and optimize the model) """
        stats = self._stats_init()
        for batch_idx, data in enumerate(self.data_loader):
            batch_start_time = time.time()
            self._setup_writer()
            data = self._data_to_device(data)
            model_output, loss, metrics = self._run_and_optimize_model(data)

            if batch_idx % self.log_step == 0:
                self._write_images(data, model_output)
                if self.verbosity >= 2:
                    self._print_log(epoch, batch_idx, batch_start_time, loss, metrics)

            products = {
                'data': data,
                'model_output': model_output,
                'loss': loss,
                'metrics': metrics
            }
            stats = self._stats_update(stats, products)
        return self._stats_finalize(stats)

    def _finalize_output(self, epoch_stats):
        """ The output of trainer and validator are logged messages. """
        epoch_time, avg_loss, avg_metrics = epoch_stats
        log = {
            'epoch_time': epoch_time,
            'avg_loss': avg_loss,
        }
        # Metrics is a list
        for i, item in enumerate(global_config['metrics'].values()):
            key = item["args"]["nickname"]
            log[f"avg_{key}"] = avg_metrics[i]

        return log

    def run(self, epoch):
        self._setup_model()
        with torch.set_grad_enabled(self.enable_grad):
            epoch_stats = self._iter_data(epoch)
        output = self._finalize_output(epoch_stats)
        return output
