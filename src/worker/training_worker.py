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
    def _init_output(self):
        """ Initialize epoch statistics like elapsed time, total loss, and metrics """
        epoch_start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.evaluation_metrics))
        return epoch_start_time, total_loss, total_metrics

    def _update_output(self, output, products):
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

    def _finalize_output(self, output):
        """ Return saved inference results along with log messages """
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

    def _get_and_write_gan_loss(self, data, model_output, network_name):
        """ Calculate GAN loss and write them to Tensorboard
        """
        loss_function = self.gan_loss_functions[network_name]
        loss = loss_function(data, model_output) * loss_function.weight
        self.writer.add_scalar(f'{loss_function.nickname}', loss.item())
        return loss
