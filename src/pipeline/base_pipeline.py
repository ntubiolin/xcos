import os
import json
import datetime
import logging
from abc import ABC, abstractmethod

import torch
import pandas as pd

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
        self.data_loader = self._setup_data_loader()
        self.valid_data_loaders = self._setup_valid_data_loaders()
        self.test_data_loaders = self._setup_test_data_loaders()

        self.optimize_strategy = global_config.get('optimize_strategy', 'normal')
        self.validation_strategy = global_config.get('validation_strategy', self.optimize_strategy)
        self._setup_model()
        self._setup_data_parallel()

        self._setup_writer()
        self.evaluation_metrics = self._setup_evaluation_metrics()

        self._setup_pipeline_specific_attributes()
        self._setup_config()

        if args.resumed_checkpoint is not None:
            self._resume_checkpoint(args.resumed_checkpoint)

        if args.pretrained is not None:
            self._load_pretrained(args.pretrained)

        self.worker_outputs = {}
        self.workers = self._create_workers()

    @abstractmethod
    def _setup_config(self):
        pass

    def _setup_pipeline_specific_attributes(self):
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

    def _get_non_parallel_model(self):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        return model

    def _setup_data_loader(self, key='data_loader'):
        return get_instance(module_data, key, global_config)

    def _setup_data_loaders(self, key):
        data_loaders = [
            getattr(module_data, entry['type'])(**entry['args'])
            for entry in global_config[key].values()
        ]
        return data_loaders

    def _setup_valid_data_loaders(self):
        if 'valid_data_loaders' in global_config.keys():
            valid_data_loaders = self._setup_data_loaders('valid_data_loaders')

            if self.data_loader.validation_split > 0:
                raise ValueError(f'Split ratio should not > 0 when other validation loaders are specified.')
        elif self.data_loader.validation_split > 0:
            valid_data_loaders = [self.data_loader.split_validation()]
        else:
            valid_data_loaders = []
        return valid_data_loaders

    def _setup_test_data_loaders(self):
        return None

    def _setup_evaluation_metrics(self):
        evaluation_metrics = [
            getattr(module_metric, entry['type'])(**entry['args']).to(self.device)
            for entry in global_config['metrics'].values()
        ]
        return evaluation_metrics

    def _setup_optimizers(self):
        """ Setup optimizers according to configuration.
            Each optimizer has its corresponding network(s) to train, specified by 'target_network' in configuraion.
            If no `target_network` is specified, all parameters of self.model will be included.
        """
        self.optimizers = {}
        for name, entry in global_config['optimizers'].items():
            model = self._get_non_parallel_model()
            if 'target_network' in entry.keys():
                network = getattr(model, entry['target_network'])
            else:
                network = model
                logger.warning(f'Target network of optimizer "{name}" not specified. '
                               f'All params of self.model will be included.')
            trainable_params = filter(lambda p: p.requires_grad, network.parameters())
            self.optimizers[name] = getattr(torch.optim, entry['type'])(trainable_params, **entry['args'])

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
        model = self._get_non_parallel_model()
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    def _resume_checkpoint(self, resumed_checkpoint):
        """
        Resume from saved resumed_checkpoints
        :param resume_path: resumed_checkpoint path to be resumed
        """
        self._resume_model_params(resumed_checkpoint)
        from .training_pipeline import TrainingPipeline
        if isinstance(self, TrainingPipeline):
            self._resume_training_state(resumed_checkpoint)
        logger.info(f"resumed_checkpoint (trained epoch {self.start_epoch - 1}) loaded")

    def _resume_training_state(self, resumed_checkpoint):
        """ States only for training pipeline like iteration counts, optimizers,
        and lr_schedulers are resumed in this function """
        self.start_epoch = resumed_checkpoint['epoch'] + 1
        self.monitor_best = resumed_checkpoint['monitor_best']

        # Estimated iteration_count is based on length of the current data loader,
        # which will be wrong if the batch sizes between the two training processes are different.
        self.train_iteration_count = resumed_checkpoint.get('train_iteration_count', 0)
        self.valid_iteration_counts = resumed_checkpoint.get(
            'valid_iteration_counts', [0] * len(self.valid_data_loaders))
        self.valid_iteration_counts = list(self.valid_iteration_counts)

        # load optimizer state from resumed_checkpoint only when optimizer type is not changed.
        optimizers_ckpt = resumed_checkpoint['optimizers']
        for key in global_config['optimizers'].keys():
            if key not in optimizers_ckpt.keys():
                logger.warning(f'Optimizer name {key} in config file is not in checkpoint (not resumed)')
            elif resumed_checkpoint['config']['optimizers'][key]['type'] != global_config['optimizers'][key]['type']:
                logger.warning(f'Optimizer type in config file is different from that of checkpoint (not resumed)')
            else:
                self.optimizers[key].load_state_dict(optimizers_ckpt[key])

    def _resume_model_params(self, resumed_checkpoint):
        """ Load model parameters from resumed checkpoint """
        # load architecture params from resumed_checkpoint.
        if resumed_checkpoint['config']['arch'] != global_config['arch']:
            logger.warning(
                'Warning: Architecture config given in config file is different from that of resumed_checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        model = self._get_non_parallel_model()
        model.load_state_dict(resumed_checkpoint['state_dict'])

    def _print_and_write_log(self, epoch, worker_outputs, write=True):
        # This function is to print out epoch summary of workers
        # and append these summary values on the summary csv file.
        if write:
            self.writer.set_step(epoch, 'epoch_average')  # TODO: See if we can use tree-structured tensorboard logging
        logger.info(f'  epoch: {epoch:d}')
        epoch_record = {'epoch': epoch}
        # print the logged info for each loader (corresponding to each worker)
        for loader_name, output in worker_outputs.items():
            log = output['log']
            if global_config.verbosity >= 1:
                logger.info(f'  {loader_name}:')
            for key, value in log.items():
                if global_config.verbosity >= 1:
                    logger.info(f'    {str(key):20s}: {value:.4f}')
                if 'elapsed_time' not in key and write:
                    value = value.item() if isinstance(value, torch.Tensor) else value
                    epoch_record[f'{loader_name}_{key}'] = [value]
                    # TODO: See if we can use tree-structured tensorboard logging
                    self.writer.add_scalar(f'{loader_name}_{key}', value)

        # concatenate summary of this epoch into 'epochs_summary.csv'
        new_df = pd.DataFrame(epoch_record)
        csv_file = os.path.join(self.saving_dir, 'epochs_summary.csv')
        df = pd.concat([pd.read_csv(csv_file), new_df]) if os.path.exists(csv_file) else new_df
        df.to_csv(csv_file, index=False)
