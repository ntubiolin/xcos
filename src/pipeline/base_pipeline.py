import os
import re
import math
import json
import logging
import datetime
import torch
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX
from utils.logging_config import logger


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self, model, losses, metrics, optimizer, resume, config,
        data_loader, valid_data_loaders, train_logger=None
    ):
        self.config = config
        self.logger = logger

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)

        self.losses = losses
        self.metrics = metrics
        self.optimizer = optimizer

        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']

        self.train_logger = train_logger

        # configuration to monitor model performance and save best
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1

        self.data_loader = data_loader
        self.valid_data_loaders = valid_data_loaders
        self.do_validation = len(self.valid_data_loaders) > 0

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], config['name'], start_time)
        # setup visualization writer instance
        writer_dir = os.path.join(config['visualization']['log_dir'], config['name'], start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, config['visualization']['tensorboardX'])

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)

        fileHandler = logging.FileHandler(os.path.join(self.checkpoint_dir, 'log.txt'))
        self.logger.addHandler(fileHandler)

        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)
        self.train_iteration_count = 0
        self.valid_iteration_counts = [0 for _ in range(len(self.valid_data_loaders))]

        if resume:
            self._resume_checkpoint(resume)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            msg = (f"Warning: The number of GPU\'s configured to use is {n_gpu_use}"
                   f"but only {n_gpu} are available on this machine.")
            self.logger.warning(msg)
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            pattern = re.compile(r'(.+)_metrics')
            for key, value in result.items():
                if key.endswith('metrics'):
                    loader_name = pattern.match(key).groups()[0]
                    log.update({f'{loader_name}_{mtr.__name__}': value[i]
                                for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            self.writer.set_step(epoch, 'epoch_average')
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))
                        if 'epoch' not in key:
                            self.writer.add_scalar(key, value)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
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
                        self.logger.warning(msg)
            if epoch % self.save_freq == 0 or best:
                self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

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
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # Estimated iteration_count is based on length of the current data loader,
        # which will be wrong if the batch sizes between the two training processes are different.
        self.train_iteration_count = checkpoint.get(
            'train_iteration_count',
            (checkpoint['epoch'] - 1) * len(self.data_loader))
        self.valid_iteration_counts = checkpoint.get(
            'valid_iteration_counts', [
                (checkpoint['epoch'] - 1) * len(self.valid_data_loaders[i])
                for i in range(len(self.valid_data_loaders))])
        self.valid_iteration_counts = list(self.valid_iteration_counts)

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. '
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def load_pretrained(self, pretrained_path):
        self.logger.info("Loading pretrained checkpoint: {} ...".format(pretrained_path))
        checkpoint = torch.load(pretrained_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
