import time

import numpy as np

from .worker_template import WorkerTemplate
from utils.global_config import global_config


class TrainingWorker(WorkerTemplate):
    """
    The middle class between WorkerTemplate and Trainer/Validator for
    trainer/validator's common processing of epoch stats and output
    finalization.
    Note:
        Inherited from WorkerTemplate.
    """
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
