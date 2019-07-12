import os
from abc import ABC, abstractmethod

import torch

from utils.logging_config import logger


class BasePipeline(ABC):
    """
    Base pipeline for training/validation/testing process
    """
    def __init__(
        self, model, data_loader, config,
        losses=None, metrics=None, optimizer=None,
        writer=None, checkpoint_dir=None,
        valid_data_loaders=[], lr_scheduler=None,
        train_logger=None
    ):
        self.config = config
        self.model = model
        self.data_loader = data_loader

        self.losses = losses
        self.metrics = metrics
        self.optimizer = optimizer
        self.valid_data_loaders = valid_data_loaders
        self.writer = writer
        self.checkpoint_dir = checkpoint_dir
        self.lr_scheduler = lr_scheduler
        self.train_logger = train_logger

        self._setup_config()
        self.workers = self._create_workers()

    @abstractmethod
    def _setup_config(self):
        pass

    @abstractmethod
    def _create_workers(self):
        return []

    def _record_log(self, epoch, log):
        # print logged informations to the screen
        self.writer.set_step(epoch, 'epoch_average')
        if self.train_logger is not None:
            self.train_logger.add_entry(log)
            if self.verbosity >= 1:
                for key, value in log.items():
                    logger.info('    {:15s}: {}'.format(str(key), value))
                    if 'epoch' not in key:
                        self.writer.add_scalar(key, value)

    def _check_and_save_best(self, epoch, log):
        """
        Evaluate model performance according to configured metric, save best checkpoint as model_best
        """
        best = False
        if self.monitor_mode != 'off':
            try:
                if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) or\
                        (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                    self.monitor_best = log[self.monitor]
                    best = True
            except KeyError:
                if epoch == 1:
                    msg = "Warning: Can\'t recognize metric named '{}' ".format(self.monitor)\
                        + "for performance monitoring. model_best checkpoint won\'t be updated."
                    logger.warning(msg)
        if epoch % self.save_freq == 0 or best:
            self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
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
            'logger': self.train_logger,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config,
            'train_iteration_count': self.train_iteration_count,
            'valid_iteration_counts': self.valid_iteration_counts,
        }

        best_str = '-best' if save_best else ''
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}_{self.monitor_best:.4f}{best_str}.pth')
        torch.save(state, filename)
        logger.info("Saving checkpoint: {} ...".format(filename))

    def run(self):
        """
        Full pipeline logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            all_logs = {}
            for worker in self.workers:
                log = worker.run(epoch)
                all_logs = {**all_logs, **log}

            self._record_log(epoch, all_logs)
            self._check_and_save_best(epoch, all_logs)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
