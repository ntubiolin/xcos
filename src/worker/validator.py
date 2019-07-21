from .worker_template import WorkerTemplate
from utils.global_config import global_config


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

    def _setup_model(self):
        self.model.eval()
