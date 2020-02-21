import time

from .worker_template import WorkerTemplate


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
        for metric in self.evaluation_metrics:
            metric.clear()
        return epoch_start_time, total_loss

    def _update_output(self, output: dict, products: dict, write_metric=True):
        """ Update epoch statistics """
        loss = products['loss']
        epoch_start_time, total_loss = output
        total_loss += loss.item()
        self._update_all_metrics(products['data'], products['model_output'], write=write_metric)
        return epoch_start_time, total_loss

    def _average_stats(self, total_loss):
        """ Calculate the average loss/metrics in this epoch """
        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = {metric.nickname: metric.finalize() for metric in self.evaluation_metrics}
        return avg_loss, avg_metrics

    def _finalize_output(self, output):
        """ Return saved inference results along with log messages """
        epoch_start_time, total_loss = output
        avg_loss, avg_metrics = self._average_stats(total_loss)
        log = {
            'elapsed_time (s)': time.time() - epoch_start_time,
            'avg_loss': avg_loss,
        }
        for key, value in avg_metrics.items():
            log[f"avg_{key}"] = value

        return {'log': log}

    def _get_and_write_gan_loss(self, data, model_output, optimize_name):
        """ Calculate GAN loss and write them to Tensorboard
        """
        loss_function = self.gan_loss_functions[optimize_name]
        loss = loss_function(data, model_output) * loss_function.weight
        self.writer.add_scalar(f'{loss_function.nickname}', loss.item())
        return loss

    def _filter_evaluation_metrics(self, metrics, scenario):
        assert scenario in ['training', 'validation']
        metrics = [metric for metric in metrics if metric.scenario == scenario]
        return metrics
