import time

import numpy as np
import torch
from torchvision.utils import make_grid

from utils.util import get_lr


class WorkerTemplate:
    def __init__(self, model, losses, metrics, optimizer, data_loader, writer):
        self.model = model
        self.losses = losses
        self.metrics = metrics
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.writer = writer

    def _setup_model(self):
        np.random.seed()
        self.model.train()
        self.logger.info(f'Current lr: {get_lr(self.optimizer)}')

    def _setup_writer(self):
        self.writer.set_step(self.step, self.data_loader.name)

    def _get_weighted_losses(self, data, model_output):
        weighted_losses = {}
        for loss_name, (loss_instance, loss_weight) in self.losses.items():
            if loss_weight <= 0.0:
                continue
            weighted_loss = loss_instance(data, model_output) * loss_weight
            weighted_losses[loss_name] = weighted_loss
        return weighted_losses

    def _eval_metrics(self, data_input, model_output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(data_input, model_output)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _run_and_optimize_model(self, data):
        self.optimizer.zero_grad()
        model_output = self.model(data)
        weighted_losses = self._get_weighted_losses(data, model_output)
        loss = sum(weighted_losses.values())
        loss.backward()
        self.optimizer.step()

        metrics = self._eval_metrics(data, model_output)
        return model_output, loss, weighted_losses, metrics

    def _data_to_device(self, data):
        for key in data.keys():
            # Dataloader yeilds something that's not tensor, e.g data['video_id']
            if torch.is_tensor(data[key]):
                    data[key] = data[key].to(self.device)
        return data

    def _write_scalers(self, loss, weighted_losses, metrics):
        for loss_name, weighted_loss in weighted_losses.items():
            self.writer.add_scalar(f'weighted_{loss_name}', weighted_loss.item())
        self.writer.add_scalar('loss', loss.item())

    def _write_images(self, data, model_output):
        self.writer.add_image("data_input", make_grid(data["data_input"], nrow=4, normalize=True))

    def _iter_data(self):
        epoch_start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):
            batch_start_time = time.time()

            self._setup_writer()

            data = self._data_to_device(data)

            model_output, loss, weighted_losses, metrics = self._run_and_optimize_model(data)

            self._write_scalers(loss, weighted_losses, metrics)
            if batch_idx % self.log_step == 0:
                self._write_images(data, model_output)
                if self.verbosity >= 2:
                    self._print_log(batch_start_time, loss, metrics)

            total_loss += loss.item()
            total_metrics += metrics

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        self._record_log(epoch_time, avg_loss, avg_metrics)

    def _setup_lr_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def run(self):
        self._setup_model()
        self._iter_data()
        self._setup_lr_scheduler()
