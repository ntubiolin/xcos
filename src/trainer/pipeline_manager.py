import os
import math
import json
import logging
import datetime

import torch

from utils.util import ensure_dir, get_instance
from utils.visualization import WriterTensorboardX
from utils.logging_config import logger
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer.trainer import Trainer
from utils.logger import Logger
from utils.logging_config import logger
import global_variables



class PipelineManager():
    """
    Training pipeline manager that controls train/validation process
    """
    def __init__(self, args, config, logger):
        self.args = args
        self.config = config
        self.logger = logger

        self.model = None
        self.data_loader = None
        self.valid_data_loaders = []
        self.loss_functions = None
        self.evaluation_metrics = None
        self.optimizer = None
        self.lr_scheduler = None

        # _setup_pipeline() will intialize the above attribute if needed, based on the config
        self.pipeline = self._setup_pipeline()

    def _setup_device(self):
        self.device, device_ids = self._prepare_device(self.config['n_gpu'])

    def _setup_model(self):
        model = get_instance(
            module_arch, 'arch', self.config,
        )
        model.summary()
        self.model = model.to(self.device)

    def _setup_data_loader(self):
        self.data_loader = get_instance(module_data, 'data_loader', self.config)

    def _setup_valid_data_loaders(self):
        if 'valid_data_loaders' in self.config.keys():
            self.valid_data_loaders = [
                getattr(module_data, entry['type'])(**entry['args'])
                for entry in self.config['valid_data_loaders']
            ]
        else:
            self.valid_data_loaders = [self.data_loader.split_validation()]

    def _setup_loss_functions(self):
        self.loss_functions = {
            entry.get('nickname', entry['type']): (
                getattr(module_loss, entry['type'])(**entry['args']),
                entry['weight']
            )
            for entry in self.config['losses']
        }

    def _setup_evaluation_metrics(self):
        self.evaluation_metrics = [
            getattr(module_metric, entry['type'])(**entry['args'])
            for entry in self.config['metrics']
        ]

    def _setup_optimizer(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', self.onfig, trainable_params)

    def _setup_lr_scheduler(self):
        self.lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', self.config, self.optimizer)

    def _setup_trainer(self):
        self.trainer = Trainer(
            self.model, self.loss_functions, self.evaluation_metrics, self.optimizer,
            resume=self.args.resume,
            config=self.config,
            data_loader=self.data_loader,
            valid_data_loaders=self.valid_data_loaders,
            lr_scheduler=self.lr_scheduler,
            **self.config['trainer_args']
        )

        if self.args.pretrained is not None:
            self.trainer.load_pretrained(self.args.pretrained)

    def _setup_tester(self):
        # this line is to solve the error described in https://github.com/pytorch/pytorch/issues/973
        torch.multiprocessing.set_sharing_strategy('file_system')
        saved_keys = ['verb_logits', 'noun_logits', 'uid', 'verb_class', 'noun_class']
        for loader in trainer.valid_data_loaders:
            file_path = os.path.join(args.save_dir, loader.name + '.pkl')
            if os.path.exists(file_path) and args.skip_exists:
                logger.warning(f'Skipping inference and saving {file_path}')
                continue
            inference_results = trainer.inference(loader, saved_keys)
            with open(file_path, 'wb') as f:
                logger.info(f'Saving results on loader {loader.name} into {file_path}')
                pickle.dump(inference_results, f)

    def run(self):
        self.pipeline.run()

    def _setup_pipeline(self, config):
        self._setup_device()
        self._setup_model()
        self._setup_data_loader()

        if self.args.mode == 'train':
            self._setup_valid_data_loaders()
            self._setup_loss_functions()
            self._setup_optimizer()
            self._setup_lr_scheduler()
            self.pipeline = self._setup_trainer()
        else:
            self.pipeline = self._setup_tester()

        self._setup_evaluation_metrics()
