import time

import numpy as np

from .training_worker import TrainingWorker
from utils.logging_config import logger
from utils.util import get_lr
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
        for attr_name in ['optimizer', 'loss_functions']:
            setattr(self, attr_name, getattr(pipeline, attr_name))

    @property
    def enable_grad(self):
        return True

    def _print_log(self, epoch, batch_idx, batch_start_time, loss, metrics):
        current_sample_idx = batch_idx * self.gt_data_loader.batch_size
        total_sample_num = self.gt_data_loader.n_samples
        sample_percentage = 100.0 * batch_idx / len(self.gt_data_loader)
        batch_time = time.time() - batch_start_time
        logger.info(
            f'Epoch: {epoch} [{current_sample_idx}/{total_sample_num} '
            f' ({sample_percentage:.0f}%)] '
            f'loss_total: {loss.item():.6f}, '
            f'BT: {batch_time:.2f}s'
        )

    def _run_and_optimize_model(self, data):
        self.optimizer.zero_grad()
        model_output = self.model(data)
        loss = self._get_and_write_loss(data, model_output)
        loss.backward()
        self.optimizer.step()

        metrics = self._get_and_write_metrics(data, model_output)
        return model_output, loss, metrics

    def _setup_model(self):
        np.random.seed()
        self.model.train()
        logger.info(f'Current lr: {get_lr(self.optimizer)}')
