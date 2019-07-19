import os
import json
import datetime
from abc import ABC, abstractmethod

import torch

from utils.util import ensure_dir, get_instance
from utils.visualization import WriterTensorboardX
from utils.logging_config import logger
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch


class BasePipeline(ABC):
    """
    Base pipeline for training/validation/testing process
    """
    def __init__(
        self, args, config
    ):
        self.config = config
        self.start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self._setup_device()
        self._setup_data_loader()

        self._setup_valid_data_loaders()

        self._setup_model_and_optimizer()
        self._setup_checkpoint_dir()
        self._setup_writer()
        self._setup_evaluation_metrics()

        self._setup_config()
        self.verbosity = config['trainer']['verbosity']

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        if args.pretrained is not None:
            self._load_pretrained(args.pretrained)

        self._worker_shared_list = {
            'common': ['config', 'device', 'model', 'loss_functions',
                       'evaluation_metrics', 'writer', 'lr_scheduler'],
            'trainer': ['optimizer'], 'tester': [], 'validator': []
        }

    @abstractmethod
    def _setup_config(self):
        pass

    @abstractmethod
    def _create_workers(self):
        return []

    # =============== functions for setting up attributes (start) ================

    def _setup_device(self):
        def prepare_device(n_gpu_use):
            """
            setup GPU device if available, move model into configured device
            """
            n_gpu = torch.cuda.device_count()
            if n_gpu_use > 0 and n_gpu == 0:
                logger.warning(
                    "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
                n_gpu_use = 0
            if n_gpu_use > n_gpu:
                msg = (f"Warning: The number of GPU\'s configured to use is {n_gpu_use} "
                       f"but only {n_gpu} are available on this machine.")
                logger.warning(msg)
                n_gpu_use = n_gpu
            device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
            list_ids = list(range(n_gpu_use))
            return device, list_ids
        self.device, self.device_ids = prepare_device(self.config['n_gpu'])

    def _setup_model_and_optimizer(self):
        """ Setup model and optimizer

        Load pretrained / resume checkpoint / data parallel if specified """
        model = get_instance(
            module_arch, 'arch', self.config,
        )
        # Print out the model architecture and number of parameters
        model.summary()
        self.model = model.to(self.device)

        self._setup_optimizer()

        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

    def _setup_data_loader(self):
        self.data_loader = get_instance(module_data, 'data_loader', self.config)

    def _setup_valid_data_loaders(self):
        if 'valid_data_loaders' in self.config.keys():
            self.valid_data_loaders = [
                getattr(module_data, entry['type'])(**entry['args'])
                for entry in self.config['valid_data_loaders']
            ]
        elif self.data_loader.validation_split > 0:
            self.valid_data_loaders = [self.data_loader.split_validation()]
        else:
            self.valid_data_loaders = []

    def _setup_evaluation_metrics(self):
        self.evaluation_metrics = [
            getattr(module_metric, entry['type'])(**entry['args'])
            for entry in self.config['metrics']
        ]

    def _setup_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', self.config, trainable_params)

    def _setup_checkpoint_dir(self):
        self.checkpoint_dir = os.path.join(self.config['trainer']['save_dir'], self.config['name'], self.start_time)
        # Save configuration file into checkpoint directory:

        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=False)

    def _setup_writer(self):
        # setup visualization writer instance
        writer_dir = os.path.join(self.config['visualization']['log_dir'], self.config['name'], self.start_time)
        self.writer = WriterTensorboardX(writer_dir, logger, self.config['visualization']['tensorboardX'])
        self.start_epoch = 1
        self.train_iteration_count = 0
        self.valid_iteration_counts = [0] * len(self.valid_data_loaders)

    # =============== functions for setting up attributes (start) ================

    def _load_pretrained(self, pretrained_path):
        """ Load pretrained model not strictly """
        logger.info("Loading pretrained checkpoint: {} ...".format(pretrained_path))
        checkpoint = torch.load(pretrained_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        logger.info("Resuming checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # Estimated iteration_count is based on length of the current data loader,
        # which will be wrong if the batch sizes between the two training processes are different.
        self.train_iteration_count = checkpoint.get('train_iteration_count', 0)
        self.valid_iteration_counts = checkpoint.get(
            'valid_iteration_counts', [0] * len(self.valid_data_loaders))
        self.valid_iteration_counts = list(self.valid_iteration_counts)

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        model = self.model.module if len(self.device_ids) > 1 else self.model
        model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. '
                           'Optimizer parameters not being resumed.')
        elif self.optimizer is None:
            logger.warning("Not loading optimizer state because it's not initialized.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch - 1))

    def _print_and_record_log(self, epoch, log):
        # print logged informations to the screen
        self.writer.set_step(epoch, 'epoch_average')
        for key, value in log.items():
            if self.verbosity >= 1:
                logger.info('    {:20s}: {:.4f}'.format(str(key), value))
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
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config,
            'train_iteration_count': self.train_iteration_count,
            'valid_iteration_counts': self.valid_iteration_counts,
        }

        best_str = '-best' if save_best else ''
        filename = os.path.join(
            self.checkpoint_dir, f'ckpt-ep{epoch}-{self.monitor}{self.monitor_best:.4f}{best_str}.pth'
        )
        torch.save(state, filename)
        logger.info("Saving checkpoint: {} ...".format(filename))

    def run(self):
        """
        Full pipeline logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            all_logs = {}
            for worker in self.workers:
                assert self.model == worker.model, f"{self.model} != {worker.model}"
                log = worker.run(epoch)
                all_logs = {**all_logs, **log}

            self._print_and_record_log(epoch, all_logs)
            self._check_and_save_best(epoch, all_logs)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
