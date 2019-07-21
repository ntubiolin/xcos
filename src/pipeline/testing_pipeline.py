from .base_pipeline import BasePipeline
from worker.tester import Tester


class TestingPipeline(BasePipeline):
    def __init__(self, args):
        #     """
        #     # this line is to solve the error described in https://github.com/pytorch/pytorch/issues/973
        #     torch.multiprocessing.set_sharing_strategy('file_system')
        #     saved_keys = ['verb_logits', 'noun_logits', 'uid', 'verb_class', 'noun_class']
        #     for loader in self.valid_data_loaders:
        #         file_path = os.path.join(self.args.save_dir, loader.name + '.pkl')
        #         if os.path.exists(file_path) and self.args.skip_exists:
        #             logger.warning(f'Skipping inference and saving {file_path}')
        #             continue
        #         inference_results = trainer.inference(loader, saved_keys)
        #         with open(file_path, 'wb') as f:
        #             logger.info(f'Saving results on loader {loader.name} into {file_path}')
        #             pickle.dump(inference_results, f)

        #     """
        super().__init__(args)
        self._create_workers()

    def _setup_config(self):
        pass

    def _create_workers(self):
        tester = Tester(self, self.data_loader, 0)
        workers = [tester]
        return workers
