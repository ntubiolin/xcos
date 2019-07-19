from .base_pipeline import BasePipeline
from worker.tester import Tester


class TestingPipeline(BasePipeline):
    def __init__(self, args, config):
        super().__init__(args, config)
        self._create_workers()

    def _setup_config(self):
        pass

    def _create_workers(self):
        tester = Tester(
            self.config, self.model, self.data_loader,
            self.loss_functions, self.evaluation_metrics, self.optimizer, self.writer
        )
        workers = [tester]
        return workers
