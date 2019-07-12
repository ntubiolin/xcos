import time
import numpy as np
import torch
from torchvision.utils import make_grid

from .worker_template import WorkerTemplate
from utils.util import get_lr
from utils.logging_config import logger


class Trainer(WorkerTemplate):
    """
    Trainer class

    Note:
        Inherited from WorkerTemplate.
    """
    def _eval_metrics(self, data_input, model_output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(data_input, model_output)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        np.random.seed()
        self.model.train()
        logger.info(f'Current lr: {get_lr(self.optimizer)}')
        epoch_start_time = time.time()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data_input in enumerate(self.data_loader):
            self.writer.set_step(self.train_iteration_count, self.data_loader.name)
            self.train_iteration_count += 1
            for key in data_input.keys():
                # Dataloader yeilds something that's not tensor, e.g data_input['video_id']
                if torch.is_tensor(data_input[key]):
                    data_input[key] = data_input[key].to(self.device)
            batch_start_time = time.time()

            self.optimizer.zero_grad()
            model_output = self.model(data_input)

            loss = self._get_loss(data_input, model_output)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('total_loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(data_input, model_output)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                logger.info(
                    f'Epoch: {epoch} [{batch_idx * self.data_loader.batch_size}/{self.data_loader.n_samples} '
                    f' ({100.0 * batch_idx / len(self.data_loader):.0f}%)] '
                    f'loss_total: {loss.item():.6f}, '
                    f'BT: {time.time() - batch_start_time:.2f}s'
                )
                self._write_images(data_input, model_output)

        log = {
            'epoch_time': time.time() - epoch_start_time,
            'loss': total_loss / len(self.data_loader),
            f'{self.data_loader.name}_metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            for idx in range(len(self.valid_data_loaders)):
                val_log = self._valid_epoch(epoch, idx)
                log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _get_loss(self, data_input, model_output):
        losses = []
        for loss_name, (loss_instance, loss_weight) in self.losses.items():
            if loss_weight <= 0.0:
                continue
            loss = loss_instance(data_input, model_output) * loss_weight
            losses.append(loss)
            self.writer.add_scalar(f'{loss_name}', loss.item())
        loss = sum(losses)
        return loss

    def _write_images(self, data_input, model_output):

        self.writer.add_image("data_input", make_grid(data_input["data_input"], nrow=4, normalize=True))
