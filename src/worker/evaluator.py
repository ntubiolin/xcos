import time

import numpy as np

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
        total_metrics = np.zeros(len(self.evaluation_metrics))
        return epoch_start_time, total_metrics

    def _update_output(self, output, metrics):
        epoch_start_time, total_metrics = output
        total_metrics += metrics
        return epoch_start_time, total_metrics

    def _finalize_output(self, output):
        epoch_start_time, total_metrics = output
        avg_metrics = (total_metrics / len(self.gt_data_loader)).tolist()
        log = {
            'elapsed_time (s)': time.time() - epoch_start_time,
        }
        # Metrics is a list
        for i, item in enumerate(global_config['metrics'].values()):
            key = item["args"]["nickname"]
            log[f"avg_{key}"] = avg_metrics[i]

        return {'log': log}

    def _to_log(self, epoch_stats):
        return {}

    def _print_log(self, epoch, batch_idx, batch_start_time, loss, metrics):
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
            metrics = self._get_and_write_metrics(gt, result, write=False)
            output = self._update_output(output, metrics)

            if batch_idx % global_config.log_step == 0:
                if global_config.verbosity >= 2:
                    self._print_log(epoch, batch_idx, batch_start_time, 0, metrics)
        return output
