import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from torchvision.utils import make_grid

from model.base_model import BaseModel
from data_loader.base_data_loader import BaseDataLoader


class WorkerTemplate(ABC):
    """ Worker template, base class for trainer, validator and tester.

    Child class need to implement at least the _run_and_optimize_model() method
    that deals with the main optimization & model inference.
    """
    def __init__(
        self, config: dict, device, model: BaseModel, data_loader: BaseDataLoader,
        losses: dict, metrics: list, optimizer,
        writer, lr_scheduler,
        **kwargs
    ):
        self.config = config
        self.device = device
        self.model = model
        self.losses = losses
        self.metrics = metrics
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.writer = writer
        self.lr_scheduler = lr_scheduler

        self.log_step = config['trainer']['log_step']
        self.verbosity = config['trainer']['verbosity']

        self.step = 0  # Tensorboard log step

    # ============ Implement the following functions ==============
    @abstractmethod
    def _run_and_optimize_model(self, data):
        """ Put data into model and optimize the model"""
        pass

    def _print_log(self, epoch, batch_idx, batch_start_time, loss, metrics):
        """ Print messages on terminal. """
        pass

    def _to_log(self, epoch, epoch_time, avg_loss, avg_metrics):
        """ Turn loss and metrics to log dict"""
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
        for loss_name, (loss_instance, loss_weight) in self.losses.items():
            if loss_weight <= 0.0:
                continue
            loss = loss_instance(data, model_output) * loss_weight
            losses.append(loss)
            self.writer.add_scalar(f'{loss_name}', loss.item())
        loss = sum(losses)
        self.writer.add_scalar('loss', loss.item())
        return loss

    def _get_and_write_metrics(self, data, model_output):
        """ Calculate evaluation metrics and write them to Tensorboard """
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(data, model_output)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _data_to_device(self, data):
        """ Put data into CPU/GPU """
        for key in data.keys():
            # Dataloader yeilds something that's not tensor, e.g data['video_id']
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(self.device)
        return data

    def _iter_data(self, epoch):
        """ Iterate through the dataset and do inference, calculate losses and metrics (and optimize the model) """
        epoch_start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):
            batch_start_time = time.time()

            self._setup_writer()

            data = self._data_to_device(data)

            model_output, loss, metrics = self._run_and_optimize_model(data)

            if batch_idx % self.log_step == 0:
                self._write_images(data, model_output)
                if self.verbosity >= 2:
                    self._print_log(epoch, batch_idx, batch_start_time, loss, metrics)

            total_loss += loss.item()
            total_metrics += metrics

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        return epoch_time, avg_loss, avg_metrics

    def _update_lr(self):
        """ Update learning rate if there is a lr_scheduler """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def run(self, epoch):
        self._setup_model()
        epoch_time, avg_loss, avg_metrics = self._iter_data(epoch)
        self._update_lr()
        log = self._to_log(epoch, epoch_time, avg_loss, avg_metrics)
        return log
