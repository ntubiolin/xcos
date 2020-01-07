import math
import os

import torch

from .base_pipeline import BasePipeline
from worker.trainer import Trainer
from worker.validator import Validator
import model.loss as module_loss
from utils.global_config import global_config
from utils.logging_config import logger
from utils.util import ensure_dir


class TrainingPipeline(BasePipeline):
    def __init__(self, args):
        super().__init__(args)

    def _setup_pipeline_specific_attributes(self):
        self._setup_loss_functions()
        if self.optimize_strategy == 'GAN':
            self._setup_gan_loss_functions()
        self._setup_optimizers()
        self._setup_lr_schedulers()

    def _create_saving_dir(self, args):
        saving_dir = os.path.join(global_config['trainer']['save_dir'], args.ckpts_subdir,
                                  global_config['name'], self.start_time)
        ensure_dir(saving_dir)

        # create a link to the resumed checkpoint as a reference
        if args.resume is not None:
            link = os.path.join(saving_dir, 'resumed_ckpt.pth')
            os.symlink(os.path.abspath(args.resume), link)

        return saving_dir

    def _setup_loss_functions(self):
        self.loss_functions = [
            getattr(module_loss, entry['type'])(**entry['args']).to(self.device)
            for key, entry in global_config['losses'].items()
        ]

    def _setup_gan_loss_functions(self):
        """ Setup GAN loss functions. Will only be called when self.optimize_strategy == 'GAN'
        The keys of gan_losses in config should have strict one-to-one mapping with names of optimizers. """
        self.gan_loss_functions = {
            key: getattr(module_loss, entry['type'])(**entry['args']).to(self.device)
            for key, entry in global_config['gan_losses'].items()
        }

    def _setup_lr_schedulers(self):
        """ Setup learning rate schedulers according to configuration. Note that the naming of
        optimizers and lr_schedulers in configuration should have a strict one-to-one mapping.
        """
        self.lr_schedulers = {}
        for optimizer_name, optimizer in self.optimizers.items():
            entry = global_config['lr_schedulers'][optimizer_name]
            self.lr_schedulers[optimizer_name] = getattr(torch.optim.lr_scheduler, entry['type'])(
                optimizer, **entry['args'])

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

    def _setup_config(self):
        self.epochs = global_config['trainer']['epochs']
        self.save_freq = global_config['trainer']['save_freq']

        # configuration to monitor model performance and save best
        self.monitored_loader = global_config['trainer']['monitored_loader']
        valid_loader_names = [loader.name for loader in self.valid_data_loaders]
        assert self.monitored_loader in valid_loader_names, \
            f"Config monitored loader '{self.monitored_loader}' is not in validation data loaders {valid_loader_names}"
        self.monitored_metric = global_config['trainer']['monitored_metric']
        valid_metric_names = [f"avg_{metric.nickname}" for metric in self.evaluation_metrics] + ["avg_loss"]
        assert self.monitored_metric in valid_metric_names, \
            f"Config monitored metric '{self.monitored_metric}' is not in valid evaluation metrics {valid_metric_names}"
        self.monitor_mode = global_config['trainer']['monitor_mode']
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        self.do_validation = len(self.valid_data_loaders) > 0

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, add '-best.pth' at the end of the best model
        """
        arch = type(self.model).__name__

        # assure that we save the model state without DataParallel module
        if isinstance(self.model, torch.nn.DataParallel):
            # get the original state out from DataParallel module
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model_state,
            'optimizers': {key: optimizer.state_dict() for key, optimizer in self.optimizers.items()},
            'monitor_best': self.monitor_best,
            'config': global_config,
            'train_iteration_count': self.train_iteration_count,
            'valid_iteration_counts': self.valid_iteration_counts,
        }

        best_str = '-best' if save_best else ''
        monitored_name = f'{self.monitored_loader}_{self.monitored_metric}'
        filename = os.path.join(
            self.saving_dir, f'ckpt-ep{epoch:04d}-{monitored_name}{self.monitor_best:.4f}{best_str}.pth'
        )
        torch.save(state, filename)
        logger.info(f"Saving checkpoint: {filename} ...")

    def _check_and_save_best(self, epoch, worker_outputs):
        """
        Evaluate model performance according to configured metric, save best checkpoint as model_best
        """
        best = False
        if self.monitor_mode != 'off':
            try:
                metric_value = worker_outputs[self.monitored_loader]['log'][self.monitored_metric]
                if (self.monitor_mode == 'min' and metric_value < self.monitor_best) or\
                        (self.monitor_mode == 'max' and metric_value > self.monitor_best):
                    self.monitor_best = metric_value
                    best = True
            except KeyError:
                if epoch == 1:
                    msg = f"Warning: Can\'t recognize metric '{self.monitored_metric}' in '{self.monitored_loader}' "\
                        + f"for performance monitoring. model_best checkpoint won\'t be updated."
                    logger.warning(msg)
        if epoch % self.save_freq == 0 or best:
            self._save_checkpoint(epoch, save_best=best)

    def _after_epoch(self, epoch, worker_outputs):
        self._print_and_write_log(epoch, worker_outputs)
        self._check_and_save_best(epoch, worker_outputs)

        if self.lr_schedulers is not None:
            for scheduler in self.lr_schedulers.values():
                scheduler.step()

    def run(self):
        """
        Full training pipeline logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            for worker in self.workers:
                worker_output = worker.run(epoch)
                self.worker_outputs[worker.data_loader.name] = worker_output
            self._after_epoch(epoch, self.worker_outputs)
