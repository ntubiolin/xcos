import time

import numpy as np

from .training_worker import TrainingWorker
from utils.logging_config import logger
from utils.util import get_lr
from utils.global_config import global_config
from pipeline.base_pipeline import BasePipeline


class Trainer(TrainingWorker):
    """
    Trainer class

    Note:
        Inherited from WorkerTemplate.
    """
    def __init__(self, pipeline: BasePipeline, *args):
        super().__init__(pipeline, *args)
        # Some shared attributes are trainer exclusive and therefore is initialized here
        shared_attrs = ['optimizers', 'loss_functions']
        shared_attrs += ['gan_loss_functions'] if self.optimize_strategy == 'GAN' else []
        for attr_name in shared_attrs:
            setattr(self, attr_name, getattr(pipeline, attr_name))
        self.evaluation_metrics = self._filter_evaluation_metrics(self.evaluation_metrics, scenario='training')

    @property
    def enable_grad(self):
        return True

    def _print_log(self, epoch, batch_idx, batch_start_time, loss):
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
            self.optimizers['default'].zero_grad()
            model_output = self.model(data)
            _, total_loss = self._get_and_write_losses(data, model_output)

            total_loss.backward()
            self.optimizers['default'].step()

        elif self.optimize_strategy == 'multitasking':
            for optimizer_name in self.optimizers.keys():
                self.optimizers[optimizer_name].zero_grad()

            model_output = self.model(data, 'normal')
            _, total_loss = self._get_and_write_losses(data, model_output)

            total_loss.backward()

            for optimizer_name in self.optimizers.keys():
                self.optimizers[optimizer_name].step()

        elif self.optimize_strategy == 'GAN':
            total_loss = 0
            for optimizer_name in self.optimizers.keys():
                self.optimizers[optimizer_name].zero_grad()
                forward_scenario = global_config['optimizers'][optimizer_name]['forward_scenario']
                model_output = self.model(data, forward_scenario)
                loss = self._get_and_write_gan_loss(data, model_output, optimizer_name)
                loss.backward()
                total_loss += loss

                self.optimizers[optimizer_name].step()

        return model_output, total_loss

    def _setup_model(self):
        np.random.seed()
        self.model.train()
        for key, optimizer in self.optimizers.items():
            logger.info(f'Current lr of optimizer {key}: {get_lr(optimizer)}')
