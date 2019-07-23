import time

import numpy as np

from .worker_template import WorkerTemplate
from utils.global_config import global_config


class TrainingWorker(WorkerTemplate):
    """
    The middle class between WorkerTemplate and Trainer/Validator for
    trainer/validator's common processing of epoch output.
    Note:
        Inherited from WorkerTemplate.
    """
    def _output_init(self):
        """ Initialize epoch statistics like elapsed time, total loss, and metrics """
        epoch_start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.evaluation_metrics))
        return epoch_start_time, total_loss, total_metrics

    def _output_update(self, output, products):
        """ Update epoch statistics """
        loss, metrics = products['loss'], products['metrics']
        epoch_start_time, total_loss, total_metrics = output
        total_loss += loss.item()
        total_metrics += metrics
        return epoch_start_time, total_loss, total_metrics

    def _average_stats(self, total_loss, total_metrics):
        """ Calculate the average loss/metrics in this epoch """
        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        return avg_loss, avg_metrics

    def _output_finalize(self, output):
        """ The output of trainer and validator are logged messages. """
        epoch_start_time, total_loss, total_metrics = output
        avg_loss, avg_metrics = self._average_stats(total_loss, total_metrics)
        log = {
            'elapsed_time (s)': time.time() - epoch_start_time,
            'avg_loss': avg_loss,
        }
        # Metrics is a list
        for i, item in enumerate(global_config['metrics'].values()):
            key = item["args"]["nickname"]
            log[f"avg_{key}"] = avg_metrics[i]

        return {'log': log}
