import time

import numpy as np

from .training_worker import TrainingWorker
from utils.logging_config import logger
from utils.util import get_lr
from pipeline.base_pipeline import BasePipeline
from utils.global_config import global_config


class Trainer(TrainingWorker):
    """
    Trainer class

    Note:
        Inherited from WorkerTemplate.
    """
    def __init__(self, pipeline: BasePipeline, *args):
        super().__init__(pipeline, *args)
        # Some shared attributes are trainer exclusive and therefore is initialized here
        for attr_name in ['optimizer', 'loss_functions', 'optimize_strategy']:
            setattr(self, attr_name, getattr(pipeline, attr_name))
        self.last_loss_D = 2.0

    @property
    def enable_grad(self):
        return True

    def _print_log(self, epoch, batch_idx, batch_start_time, loss, metrics):
        current_sample_idx = batch_idx * self.data_loader.batch_size
        total_sample_num = self.data_loader.n_samples
        sample_percentage = 100.0 * batch_idx / len(self.data_loader)
        batch_time = time.time() - batch_start_time
        logger.info(
            f'Epoch: {epoch} [{current_sample_idx}/{total_sample_num} '
            f' ({sample_percentage:.0f}%)] '
            f'loss_total: {loss.item():.6f}, '
            f'BT: {batch_time:.2f}s'
        )

    def _run_and_optimize_model(self, data):
        if self.optimize_strategy == 'normal':
            self.optimizer['all'].zero_grad()
            model_output = self.model(data)
            losses, total_loss = self._get_and_write_losses(data, model_output)

            total_loss.backward()
            self.optimizer['all'].step()

        elif self.optimize_strategy == 'GAN':
            if (self.last_loss_D > global_config['last_d_lower_bound']
                    and data['batch_idx'] % global_config['train_d_every'] == 0):
                self.optimizer['D'].zero_grad()
                model_output = self.model(data)
                losses, total_loss = self._get_and_write_losses(data, model_output)

                loss_D = sum([loss for key, loss in losses.items() if 'discriminator' in key])
                loss_D.backward()
                self.optimizer['D'].step()

            self.optimizer['G'].zero_grad()
            model_output = self.model(data)
            losses, total_loss = self._get_and_write_losses(data, model_output)

            loss_G = sum([loss for key, loss in losses.items() if 'discriminator' not in key])
            loss_G.backward()
            self.optimizer['G'].step()

            loss_D = sum([loss for key, loss in losses.items() if 'discriminator' in key])
            self.last_loss_D = loss_D.item()

        metrics = self._get_and_write_metrics(data, model_output)
        return model_output, total_loss, metrics

    def _setup_model(self):
        np.random.seed()
        self.model.train()
        for key, optimizer in self.optimizer.items():
            logger.info(f'Current lr of optimizer {key}: {get_lr(optimizer)}')
