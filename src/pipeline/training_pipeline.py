import math

import torch

from .base_pipeline import BasePipeline
from worker.trainer import Trainer
from worker.validator import Validator
import model.loss as module_loss
from utils.util import get_instance
from utils.global_config import global_config


class TrainingPipeline(BasePipeline):
    def __init__(self, args):
        super().__init__(args)
        self._setup_loss_functions()
        self._setup_lr_scheduler()
        self.workers = self._create_workers()

    def _setup_loss_functions(self):
        self.loss_functions = [
            getattr(module_loss, entry['type'])(**entry['args'])
            for key, entry in global_config['losses'].items()
        ]

    def _setup_lr_scheduler(self):
        self.lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', global_config, self.optimizer)

    def _setup_config(self):
        self.epochs = global_config['trainer']['epochs']
        self.save_freq = global_config['trainer']['save_freq']

        # configuration to monitor model performance and save best
        self.monitored_loader = global_config['trainer']['monitored_loader']
        self.monitored_metric = global_config['trainer']['monitored_metric']
        self.monitor_mode = global_config['trainer']['monitor_mode']
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        self.do_validation = len(self.valid_data_loaders) > 0

    def _create_workers(self):
        trainer = Trainer(
            self, self.data_loader, self.train_iteration_count
        )
        workers = [trainer]

        for i, valid_data_loader in enumerate(self.valid_data_loaders):
            workers.append(
                Validator(
                    self, valid_data_loader, self.valid_iteration_counts[i]
                )
            )
        return workers
