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
        global_config.setup(args.template_config, args.specified_configs, args.resumed_checkpoint)
        self.start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.saving_dir = self._create_saving_dir(args)
        self._add_logging_file_handler()
        self._save_config_file()
        self._print_config_messages()

        self.device, self.device_ids = self._setup_device()
        self._setup_data_loader()
        self._setup_valid_data_loaders()

        self.optimize_strategy = global_config.get('optimize_strategy', 'normal')
        self._setup_model()
        self._setup_optimizers()
        self._setup_data_parallel()

        self._setup_writer()
        self.evaluation_metrics = self._setup_evaluation_metrics()

        self._setup_config()

        if args.resumed_checkpoint is not None:
            self._resume_checkpoint(args.resumed_checkpoint)

        if args.pretrained is not None:
            self._load_pretrained(args.pretrained)

        self.worker_outputs = {}
        self._before_create_workers()
        self.workers = self._create_workers()

    @abstractmethod
    def _setup_config(self):
        pass

    def _before_create_workers(self):
        pass

    @abstractmethod
    def _create_workers(self):
        return []

    # =============== functions for setting up attributes (start) ================

    @abstractmethod
    def _create_saving_dir(self, resume_path):
        """ Create directory to save ckpt, config, and logges messags. Return the created path """
        pass

    def _save_config_file(self):
        # Save configuration file into checkpoint directory
        config_save_path = os.path.join(self.saving_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(global_config, handle, indent=4, sort_keys=False)

    def _add_logging_file_handler(self):
        fileHandler = logging.FileHandler(os.path.join(self.saving_dir, 'log.txt'))
        logger.addHandler(fileHandler)

    def _print_config_messages(self):
        global_config.print_changed()
        logger.info(f'Experiment name: {global_config["name"]}')

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
        device, device_ids = prepare_device(global_config['n_gpu'])
        return device, device_ids

    def _setup_model(self):
        """ Setup model and print summary """
        model = get_instance(
            module_arch, 'arch', global_config,
        )
        # Print out the model architecture and number of parameters
        model.summary()
        self.model = model.to(self.device)

    def _setup_data_parallel(self):
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

    def _setup_data_loader(self, key='data_loader'):
        self.data_loader = get_instance(module_data, key, global_config)

    def _setup_data_loaders(self, key):
        data_loaders = [
            getattr(module_data, entry['type'])(**entry['args'])
            for entry in global_config[key].values()
        ]
        return data_loaders

    def _setup_valid_data_loaders(self):
        if 'valid_data_loaders' in global_config.keys():
            self.valid_data_loaders = self._setup_data_loaders('valid_data_loaders')

            if self.data_loader.validation_split > 0:
                raise ValueError(f'Split ratio should not > 0 when other validation loaders are specified.')
        elif self.data_loader.validation_split > 0:
            self.valid_data_loaders = [self.data_loader.split_validation()]
        else:
            self.valid_data_loaders = []

    def _setup_evaluation_metrics(self):
        evaluation_metrics = [
            getattr(module_metric, entry['type'])(**entry['args']).to(self.device)
            for entry in global_config['metrics'].values()
        ]
        return evaluation_metrics

    def _setup_optimizers(self):
        self.optimizers = {}
        for network_name in self.model._modules.keys():
            trainable_params = filter(lambda p: p.requires_grad, getattr(self.model, network_name).parameters())
            optimizer_name = f'optimizer_{network_name}'
            if optimizer_name not in global_config:
                logger.warning(f"{optimizer_name} not in global_config; using default optimizer for {network_name}")
                optimizer_name = 'optimizer'
            self.optimizers[network_name] = get_instance(torch.optim, optimizer_name, global_config, trainable_params)

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
        logger.info(f"Loading pretrained checkpoint: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path)
        model = self.model.module if len(self.device_ids) > 1 else self.model
        model.load_state_dict(checkpoint['state_dict'], strict=False)

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
        elif self.optimizers is None:
            logger.warning("Not loading optimizer state because it's not initialized.")
        else:
            for key, optimizer_state in resumed_checkpoint['optimizers'].items():
                self._setup_optimizers[key].load_state_dict(optimizer_state)

        logger.info(f"resumed_checkpoint (trained epoch {self.start_epoch - 1}) loaded")

    def _print_and_write_log(self, epoch, worker_outputs, write=True):
        # print common worker logged info
        if write:
            self.writer.set_step(epoch, 'epoch_average')  # TODO: See if we can use tree-structured tensorboard logging
        logger.info(f'  epoch: {epoch:d}')
        # print the logged info for each loader (corresponding to each worker)
        for loader_name, output in worker_outputs.items():
            log = output['log']
            if global_config.verbosity >= 1:
                logger.info(f'  {loader_name}:')
            for key, value in log.items():
                if global_config.verbosity >= 1:
                    logger.info(f'    {str(key):20s}: {value:.4f}')
                if 'elapsed_time' not in key and write:
                    # TODO: See if we can use tree-structured tensorboard logging
                    self.writer.add_scalar(f'{loader_name}_{key}', value)
