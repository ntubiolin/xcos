import math

from .base_pipeline import BasePipeline
from worker.trainer import Trainer
from worker.validator import Validator


class TrainingPipeline(BasePipeline):
    def __init__(self, args, config):
        self.config = config
        super().__init__(args, config)
        self._setup_loss_functions()
        self._setup_lr_scheduler()
        self.workers = self._create_workers()

    def _setup_config(self):
        self.epochs = self.config['trainer']['epochs']
        self.save_freq = self.config['trainer']['save_freq']

        # configuration to monitor model performance and save best
        self.monitor = self.config['trainer']['monitor']
        self.monitor_mode = self.config['trainer']['monitor_mode']
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        self.do_validation = len(self.valid_data_loaders) > 0

    def _create_workers(self):
        trainer = Trainer(
            self.config, self.device, self.model, self.data_loader,
            self.loss_functions, self.evaluation_metrics, self.optimizer,
            self.writer, self.lr_scheduler, self.train_iteration_count
        )
        workers = [trainer]
        for i, valid_data_loader in enumerate(self.valid_data_loaders):
            workers.append(
                Validator(
                    self.config, self.device, self.model, valid_data_loader,
                    self.loss_functions, self.evaluation_metrics, self.optimizer,
                    self.writer, self.lr_scheduler, self.valid_iteration_counts[i]
                )
            )
        return workers
