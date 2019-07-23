import os

import numpy as np

from .base_pipeline import BasePipeline
from worker.tester import Tester

from utils.global_config import global_config
from utils.util import ensure_dir
from utils.logging_config import logger


class TestingPipeline(BasePipeline):
    def __init__(self, args):
        """
        # You may need this line to solve the error described in https://github.com/pytorch/pytorch/issues/973
        torch.multiprocessing.set_sharing_strategy('file_system')
        """
        super().__init__(args)
        self.saved_keys = args.saved_keys
        self.workers = self._create_workers()

    def _setup_saving_dir(self, args):
        self.saving_dir = os.path.join(global_config['trainer']['save_dir'], 'outputs',
                                       global_config['name'])
        if os.path.exists(self.saving_dir):
            logger.warning('The saving directory already exists. If continued, some files might be overwriten.')
            response = input('Proceed? [y/N]')
            if response != 'y':
                logger.info('Exit.')
                exit()
        ensure_dir(self.saving_dir)
        if args.resume is not None:
            link = os.path.join(self.saving_dir, 'resumed_ckpt.pth')
            if os.path.exists(link):
                os.remove(link)
            # Mark the used resume path by a symbolic link
            os.symlink(os.path.abspath(args.resume), link)

    def _setup_config(self):
        pass

    def _create_workers(self):
        tester = Tester(self, self.data_loader, 0)
        workers = [tester]
        return workers

    def _save_inference_results(self, name: str, worker_output: dict):
        path = os.path.join(self.saving_dir, f'{name}_output.npz')
        logger.info(f'Saving {path}...')
        np.savez(path, **worker_output)

    def run(self):
        """
        Full testing pipeline logic
        """
        worker_outputs = {}
        for worker in self.workers:
            worker_output = worker.run(0)
            self._save_inference_results(worker.data_loader.name, worker_output['saved'])
            worker_outputs[worker.data_loader.name] = worker_output
        self._print_and_record_log(0, worker_outputs)
