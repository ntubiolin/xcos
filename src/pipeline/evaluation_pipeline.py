from .base_pipeline import BasePipeline
from worker.evaluator import Evaluator
from utils.global_config import global_config


class EvaluationPipeline(BasePipeline):
    def __init__(self, args):
        global_config.setup(args.template_config, args.specified_configs, args.resumed_checkpoint)
        self._print_config_messages()
        self.gt_data_loaders = self._setup_data_loaders('gt_data_loaders')
        self.result_data_loaders = self._setup_data_loaders('result_data_loaders')
        self.device, self.device_ids = self._setup_device()
        self.evaluation_metrics = self._setup_evaluation_metrics()
        self.workers = self._create_workers()
        self.worker_outputs = {}

    def _setup_config(self):
        pass

    def _setup_saving_dir(self, resume_path):
        pass

    def _create_saving_dir(self, resume_path):
        pass

    def _create_workers(self):
        workers = []
        # Add a evaluator for each data loader
        for gt_data_loader, result_data_loader in zip(self.gt_data_loaders, self.result_data_loaders):
            evaluator = Evaluator(self, gt_data_loader, result_data_loader)
            workers += [evaluator]
        return workers

    def run(self):
        """
        Full evaluation pipeline logic
        """
        for worker in self.workers:
            worker_output = worker.run(0)
            self.worker_outputs[worker.result_data_loader.name] = worker_output
        self._print_and_write_log(0, self.worker_outputs, write=False)
