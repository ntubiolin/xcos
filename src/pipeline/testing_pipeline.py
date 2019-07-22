import os

from .base_pipeline import BasePipeline
from worker.tester import Tester

from utils.global_config import global_config
from utils.util import ensure_dir


class TestingPipeline(BasePipeline):
    def __init__(self, args):
        #     """
        #     # this line is to solve the error described in https://github.com/pytorch/pytorch/issues/973
        #     torch.multiprocessing.set_sharing_strategy('file_system')
        #     """
        super().__init__(args)
        self.saved_keys = args.saved_keys
        self.workers = self._create_workers()

    def _setup_saving_dir(self, resume_path):
        self.saving_dir = os.path.join(global_config['trainer']['save_dir'], 'outputs',
                                       global_config['name'], self.start_time)
        ensure_dir(self.saving_dir)
        if resume_path is not None:
            # Mark the used resume path by a symbolic link
            os.symlink(resume_path, os.path.join(self.saving_dir, 'resumed_ckpt.pth'))

    def _setup_config(self):
        pass

    def _create_workers(self):
        tester = Tester(self, self.data_loader, 0)
        workers = [tester]
        return workers

    def _save_inference_results(self, worker_output):
        pass

    def run(self):
        """
        Full testing pipeline logic
        """
        for worker in self.workers:
            worker_output = worker.run(0)
            self._save_inference_results(worker_output)
