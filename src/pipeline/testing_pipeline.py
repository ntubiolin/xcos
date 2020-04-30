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

    def _create_saving_dir(self, args):
        saving_dir = os.path.join(global_config['trainer']['save_dir'], args.outputs_subdir,
                                  global_config['name'])
        if os.path.exists(saving_dir):
            logger.warning(f'The saving directory "{saving_dir}" already exists. '
                           f'If continued, some files might be overwriten.')
            response = input('Proceed? [y/N] ')
            if response != 'y':
                logger.info('Exit.')
                exit()
        ensure_dir(saving_dir)
        if args.resume is not None:
            link = os.path.join(saving_dir, 'resumed_ckpt.pth')
            if os.path.exists(link):
                os.remove(link)
            # Mark the used resume path by a symbolic link
            os.symlink(os.path.abspath(args.resume), link)
        return saving_dir

    def _setup_data_loader(self):
        return None

    def _setup_valid_data_loaders(self):
        return []

    def _setup_config(self):
        pass

    def _create_workers(self):
        workers = []
        # Add a tester for each data loader
        for test_data_loader in self.test_data_loaders:
            tester = Tester(pipeline=self, test_data_loader=test_data_loader)
            workers += [tester]
        return workers

    def _save_inference_results(self, name: str, worker_output: dict):
        path = os.path.join(self.saving_dir, f'{name}_output.npz')
        logger.info(f'Saving {path} ...')
        np.savez(path, **worker_output)

    def _setup_test_data_loaders(self):
        if 'test_data_loaders' in global_config.keys():
            test_data_loaders = self._setup_data_loaders('test_data_loaders')
            return test_data_loaders
        else:
            raise ValueError(f"No test_data_loaders key in config")

    def run(self):
        """
        Full testing pipeline logic
        """
        for worker in self.workers:
            worker_output = worker.run(0)
            if not global_config.save_while_infer:
                self._save_inference_results(worker.data_loader.name, worker_output['saved'])
            self.worker_outputs[worker.data_loader.name] = worker_output
        self._print_and_write_log(0, self.worker_outputs, write=True)
