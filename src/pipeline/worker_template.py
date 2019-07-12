import time

import numpy as np
import torch
from torchvision.utils import make_grid

from utils.util import get_lr
from utils.logging_config import logger


class WorkerTemplate:
    def __init__(self, config, model, data_loader, losses, metrics, optimizer, writer, log_step, **kwargs):
        self.config = config
        self.model = model
        self.losses = losses
        self.metrics = metrics
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.writer = writer
        self.log_step = log_step

    def _setup_model(self):
        np.random.seed()
        self.model.train()
        logger.info(f'Current lr: {get_lr(self.optimizer)}')

    def _setup_writer(self):
        self.writer.set_step(self.step, self.data_loader.name)

    def _get_and_write_loss(self, data, model_output):
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
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(data, model_output)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _run_and_optimize_model(self, data):
        self.optimizer.zero_grad()
        model_output = self.model(data)
        loss = self._get_and_write_loss(data, model_output)
        loss.backward()
        self.optimizer.step()

        metrics = self._get_and_write_metrics(data, model_output)
        return model_output, loss, metrics

    def _data_to_device(self, data):
        for key in data.keys():
            # Dataloader yeilds something that's not tensor, e.g data['video_id']
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(self.device)
        return data

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

            model_output, loss, metrics = self._run_and_optimize_model(data)

            if batch_idx % self.log_step == 0:
                self._write_images(data, model_output)
                if self.verbosity >= 2:
                    self._print_log(batch_start_time, loss, metrics)

            total_loss += loss.item()
            total_metrics += metrics

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        return epoch_time, avg_loss, avg_metrics

    def _to_log(self, epoch, epoch_time, avg_loss, avg_metrics):
        log = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'avg_loss': avg_loss,
            'avg_metrics': avg_metrics
        }
        return log

    def _setup_lr_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def run(self, epoch):
        self._setup_model()
        epoch_time, avg_loss, avg_metrics = self._iter_data()
        self._setup_lr_scheduler()
        log = self._to_log(epoch, epoch_time, avg_loss, avg_metrics)
        return log
