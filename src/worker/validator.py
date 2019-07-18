from .worker_template import WorkerTemplate


class Validator(WorkerTemplate):
    """
    Validator class

    Note:
        Inherited from WorkerTemplate.
    """
    def _run_and_optimize_model(self, data):
        model_output = self.model(data)
        loss = self._get_and_write_loss(data, model_output)
        metrics = self._get_and_write_metrics(data, model_output)
        return model_output, loss, metrics

    def _to_log(self, epoch, epoch_time, avg_loss, avg_metrics):
        log = {
            'valid_epoch_time': epoch_time,
            'valid_avg_loss': avg_loss,
        }
        # Metrics is a list
        for i, item in enumerate(self.config['metrics']):
            key = item["args"]["nickname"]
            log[f"valid_avg_{key}"] = avg_metrics[i]

        return log

    def _setup_model(self):
        self.model.eval()
