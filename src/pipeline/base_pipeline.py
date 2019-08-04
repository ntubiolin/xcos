import os
import json
import datetime
import logging
from abc import ABC, abstractmethod

import torch

from utils.util import get_instance
from utils.visualization import WriterTensorboard
from utils.logging_config import logger
from utils.global_config import global_config
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch


class BasePipeline(ABC):
    """
    Base pipeline for training/validation/testing process
    """
    def __init__(
        self, args
    ):
        self.start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self._setup_saving_dir(args)
        self._save_config_file()
        self._add_logging_file_handler()

        self._setup_device()
        self._setup_data_loader()
        self._setup_valid_data_loaders()

        self._setup_model_and_optimizer()

        self._setup_writer()
        self._setup_evaluation_metrics()

        self._setup_config()

        if args.resumed_checkpoint is not None:
            self._resume_checkpoint(args.resumed_checkpoint)

        if args.pretrained is not None:
            self._load_pretrained(args.pretrained)

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
        self.device, self.device_ids = prepare_device(global_config['n_gpu'])

    def _setup_model_and_optimizer(self):
        """ Setup model and optimizer

        Load pretrained / resume checkpoint / data parallel if specified """
        model = get_instance(
            module_arch, 'arch', global_config,
        )
        # Print out the model architecture and number of parameters
        model.summary()
        self.model = model.to(self.device)

        self._setup_optimizer()

        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

    def _setup_data_loader(self):
        self.data_loader = get_instance(module_data, 'data_loader', global_config)

    def _setup_valid_data_loaders(self):
        if 'valid_data_loaders' in global_config.keys():
            self.valid_data_loaders = [
                getattr(module_data, entry['type'])(**entry['args'])
                for entry in global_config['valid_data_loaders']
            ]
            if self.data_loader.validation_split > 0:
                raise ValueError(f'Split ratio should not > 0 when other validation loaders are specified.')
        elif self.data_loader.validation_split > 0:
            self.valid_data_loaders = [self.data_loader.split_validation()]
        else:
            self.valid_data_loaders = []

    def _setup_evaluation_metrics(self):
        self.evaluation_metrics = [
            getattr(module_metric, entry['type'])(**entry['args']).to(self.device)
            for entry in global_config['metrics'].values()
        ]

    def _setup_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', global_config, trainable_params)

    @abstractmethod
    def _setup_saving_dir(self, resume_path):
        pass

    def _save_config_file(self):
        # Save configuration file into checkpoint directory
        config_save_path = os.path.join(self.saving_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(global_config, handle, indent=4, sort_keys=False)

    def _add_logging_file_handler(self):
        fileHandler = logging.FileHandler(os.path.join(self.saving_dir, 'log.txt'))
        logger.addHandler(fileHandler)

    def _setup_writer(self):
        # setup visualization writer instance
        writer_dir = os.path.join(global_config['visualization']['log_dir'], global_config['name'], self.start_time)
        self.writer = WriterTensorboard(writer_dir, logger, global_config['visualization']['tensorboardX'])
        self.start_epoch = 1
        self.train_iteration_count = 0
        self.valid_iteration_counts = [0] * len(self.valid_data_loaders)

    # =============== functions for setting up attributes (start) ================

    def _load_pretrained(self, pretrained_path):
        """ Load pretrained model not strictly """
        logger.info("Loading pretrained checkpoint: {} ...".format(pretrained_path))
        checkpoint = torch.load(pretrained_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def _resume_checkpoint(self, resumed_checkpoint):
        """
        Resume from saved resumed_checkpoints

        :param resume_path: resumed_checkpoint path to be resumed
        """
        self.start_epoch = resumed_checkpoint['epoch'] + 1
        self.monitor_best = resumed_checkpoint['monitor_best']

        # Estimated iteration_count is based on length of the current data loader,
        # which will be wrong if the batch sizes between the two training processes are different.
        self.train_iteration_count = resumed_checkpoint.get('train_iteration_count', 0)
        self.valid_iteration_counts = resumed_checkpoint.get(
            'valid_iteration_counts', [0] * len(self.valid_data_loaders))
        self.valid_iteration_counts = list(self.valid_iteration_counts)

        # load architecture params from resumed_checkpoint.
        if resumed_checkpoint['config']['arch'] != global_config['arch']:
            logger.warning(
                'Warning: Architecture config given in config file is different from that of resumed_checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        model = self.model.module if len(self.device_ids) > 1 else self.model
        model.load_state_dict(resumed_checkpoint['state_dict'])

        # load optimizer state from resumed_checkpoint only when optimizer type is not changed.
        if resumed_checkpoint['config']['optimizer']['type'] != global_config['optimizer']['type']:
            logger.warning('Warning: Optimizer type given in config file is different from that of resumed_checkpoint. '
                           'Optimizer parameters not being resumed.')
        elif self.optimizer is None:
            logger.warning("Not loading optimizer state because it's not initialized.")
        else:
            self.optimizer.load_state_dict(resumed_checkpoint['optimizer'])

        logger.info(f"resumed_checkpoint (trained epoch {self.start_epoch - 1}) loaded")

    def _print_and_record_log(self, epoch, worker_outputs):
        # print common worker logged info
        self.writer.set_step(epoch, 'epoch_average')  # TODO: See if we can use tree-structured tensorboard logging
        logger.info(f'  epoch: {epoch:d}')
        # print the logged info for each loader (corresponding to each worker)
        for loader_name, output in worker_outputs.items():
            log = output['log']
            if global_config['trainer']['verbosity'] >= 1:
                logger.info(f'  {loader_name}:')
            for key, value in log.items():
                if global_config['trainer']['verbosity'] >= 1:
                    logger.info(f'    {str(key):20s}: {value:.4f}')
                if 'elapsed_time' not in key:
                    # TODO: See if we can use tree-structured tensorboard logging
                    self.writer.add_scalar(f'{loader_name}_{key}', value)
