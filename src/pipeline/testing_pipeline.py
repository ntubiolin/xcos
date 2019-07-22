from .base_pipeline import BasePipeline
from worker.tester import Tester


class TestingPipeline(BasePipeline):
    def __init__(self, args):
        #     """
        #     # this line is to solve the error described in https://github.com/pytorch/pytorch/issues/973
        #     torch.multiprocessing.set_sharing_strategy('file_system')
        #     """
        super().__init__(args)
        self.saved_keys = args.saved_keys
        self.workers = self._create_workers()

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
            breakpoint()
            self._save_inference_results(worker_output)
