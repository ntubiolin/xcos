import math

from .base_pipeline import BasePipeline
from .trainer import Trainer
from .validator import Validator


class TrainingPipeline(BasePipeline):

    def _setup_config(self):
        self.epochs = self.config['trainer']['epochs']
        self.save_freq = self.config['trainer']['save_freq']
        self.verbosity = self.config['trainer']['verbosity']

        # configuration to monitor model performance and save best
        self.monitor = self.config['trainer']['monitor']
        self.monitor_mode = self.config['trainer']['monitor_mode']
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        self.do_validation = len(self.valid_data_loaders) > 0

        self.start_epoch = 1
        self.train_iteration_count = 0
        self.valid_iteration_counts = [0 for _ in range(len(self.valid_data_loaders))]

    def _create_workers(self):
        trainer = Trainer(
            self.config, self.model, self.data_loader,
            self.losses, self.metrics, self.optimizer, self.writer, self.log_step
        )
        validator = Validator(
            self.config, self.model, self.data_loader,
            self.losses, self.metrics, self.optimizer, self.writer, self.log_step
        )
        workers = [trainer, validator]
        return workers
