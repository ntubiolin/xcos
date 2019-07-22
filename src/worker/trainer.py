import time

import numpy as np

from .worker_template import WorkerTemplate
from utils.logging_config import logger
from utils.util import get_lr
from pipeline.base_pipeline import BasePipeline


class Trainer(WorkerTemplate):
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
        self.enable_grad = True

    def _print_log(self, epoch, batch_idx, batch_start_time, loss, metrics):
        logger.info(
            f'Epoch: {epoch} [{batch_idx * self.data_loader.batch_size}/{self.data_loader.n_samples} '
            f' ({100.0 * batch_idx / len(self.data_loader):.0f}%)] '
            f'loss_total: {loss.item():.6f}, '
            f'BT: {time.time() - batch_start_time:.2f}s'
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
