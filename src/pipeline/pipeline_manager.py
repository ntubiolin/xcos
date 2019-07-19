import os
import json
import datetime

import torch

from utils.util import ensure_dir, get_instance
from utils.visualization import WriterTensorboardX
from utils.logging_config import logger
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from .training_pipeline import TrainingPipeline
from .testing_pipeline import TestingPipeline


class PipelineManager():
    """
    Training/testing pipeline manager that controls train/validation/testing process

    Example:
        pipeline_manager = PipelineManager(args, config)
        pipeline_manager.set_mode(args.mode)
        pipeline_manager.setup_pipeline()
        pipeline_manager.run()
    """
    def __init__(self, args: dict, config: dict):
        self.start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')

        self.args = args
        self.config = config

        # setup_pipeline() will intialize the following attribute if needed, based on the config
        self.model = None
        self.data_loader = None
        self.valid_data_loaders = None
        self.loss_functions = None
        self.evaluation_metrics = None
        self.optimizer = None
        self.lr_scheduler = None

        self.start_epoch = None
        self.train_iteration_count = None
        self.valid_iteration_counts = None

    def set_mode(self, mode):
        self.mode = mode

    def setup_pipeline(self):
        if self.mode == 'train':
            self._setup_loss_functions()
            self._setup_lr_scheduler()
            self.pipeline = self._create_training_pipeline()
        else:
            self.pipeline = self._create_testing_pipeline()

    def run(self):
        self.pipeline.run()
