import time

from .worker_template import WorkerTemplate
from pipeline.base_pipeline import BasePipeline
from data_loader.base_data_loader import BaseDataLoader
from utils.logging_config import logger
from utils.global_config import global_config


class Evaluator(WorkerTemplate):
    """
    Evaluator class

    Note:
        Inherited from WorkerTemplate.
    """
    def __init__(
        self,
        pipeline: BasePipeline,
        gt_data_loader: BaseDataLoader,
        result_data_loader: BaseDataLoader,
        *args
    ):
        # Attributes listed below are shared from pipeline among all different workers.
        for attr_name in ['device', 'evaluation_metrics']:
            setattr(self, attr_name, getattr(pipeline, attr_name))
        self.gt_data_loader = gt_data_loader
        self.result_data_loader = result_data_loader

    @property
    def enable_grad(self):
        return False

    def _run_and_optimize_model(self, data_input):
        return {}, None, []

    def _setup_model(self):
        pass

    def _init_output(self):
        epoch_start_time = time.time()
        for metric in self.evaluation_metrics:
            metric.clear()
        return epoch_start_time

    def _update_output(self, output, batch_products, write_metric=True):
        epoch_start_time = output
        self._update_all_metrics(batch_products['gt'], batch_products['result'], write=write_metric)
        return epoch_start_time

    def _finalize_output(self, output):
        epoch_start_time = output
        avg_metrics = {metric.nickname: metric.finalize() for metric in self.evaluation_metrics}
        log = {
            'elapsed_time (s)': time.time() - epoch_start_time,
        }
        for key, value in avg_metrics.items():
            log[f"avg_{key}"] = value
        return {'log': log}

    def _to_log(self, epoch_stats):
        return {}

    def _print_log(self, epoch, batch_idx, batch_start_time, loss):
        current_sample_idx = batch_idx * self.gt_data_loader.batch_size
        total_sample_num = self.gt_data_loader.n_samples
        sample_percentage = 100.0 * batch_idx / len(self.gt_data_loader)
        batch_time = time.time() - batch_start_time
        logger.info(
            f'Epoch: {epoch} [{current_sample_idx}/{total_sample_num} '
            f' ({sample_percentage:.0f}%)] '
            f'BT: {batch_time:.2f}s'
        )

    def _iter_data(self, epoch):
        output = self._init_output()
        for batch_idx, (gt, result) in enumerate(zip(self.gt_data_loader, self.result_data_loader)):
            batch_start_time = time.time()
            gt = self._data_to_device(gt)
            result = self._data_to_device(result)
            batch_products = {'gt': gt, 'result': result}
            output = self._update_output(output, batch_products, write_metric=False)

            if batch_idx % global_config.log_step == 0:
                if global_config.verbosity >= 2:
                    self._print_log(epoch, batch_idx, batch_start_time, 0)
        return output
