import os
import time
import numpy as np

import torch

from .worker_template import WorkerTemplate
from data_loader.base_data_loader import BaseDataLoader
from pipeline.base_pipeline import BasePipeline
from utils.global_config import global_config
from utils.logging_config import logger


class Tester(WorkerTemplate):
    """
    Tester class

    Note:
        Inherited from WorkerTemplate.
    """
    def __init__(self, pipeline: BasePipeline, test_data_loader: BaseDataLoader):
        super().__init__(pipeline=pipeline, data_loader=test_data_loader, step=0)
        for attr_name in ['saving_dir']:
            setattr(self, attr_name, getattr(pipeline, attr_name))

    @property
    def enable_grad(self):
        return False

    def _run_and_optimize_model(self, data):
        with torch.no_grad():
            model_output = self.model(data)
        return model_output, None

    def _setup_model(self):
        self.model.eval()

    def _to_log(self, epoch_stats):
        return {}

    def _init_output(self):
        """ Initialize a dictioary structure to save inferenced results. """
        return {
            'epoch_start_time': time.time(),
            'saved': {k: [] for k in global_config.saved_keys}
        }

    def _update_output(self, epoch_output, products):
        """ Update the dictionary saver: extend entries """

        def update_epoch_output_from_dict(dictionary):
            for key in dictionary.keys():
                if key not in global_config.saved_keys:
                    continue
                value = dictionary[key]
                saved_value = value.cpu().numpy() if torch.is_tensor(value) else value
                epoch_output['saved'][key].extend([v for v in saved_value])

        data, model_output = products['data'], products['model_output']
        if global_config.save_while_infer:
            # Clean previous results
            epoch_output['saved'] = {k: [] for k in global_config.saved_keys}

        for d in [data, model_output]:
            update_epoch_output_from_dict(d)

        if global_config.save_while_infer:
            # Save results
            name = self.data_loader.name

            for i in range(len(epoch_output['saved']['model_output'])):
                index = epoch_output['saved']['index'][i]
                output = {}
                for saved_key in epoch_output['saved'].keys():
                    output[saved_key] = epoch_output['saved'][saved_key][i]
                output_path = os.path.join(self.saving_dir, f'{name}_index{index:06d}.npz')
                np.savez(output_path, **output)
                if index % 1000 == 0:
                    logger.info(f'Saving output {output_path} ...')

        return epoch_output

    def _finalize_output(self, epoch_output):
        """ Return saved inference results along with log messages """
        log = {'elasped_time (s)': time.time() - epoch_output['epoch_start_time']}
        return {'saved': epoch_output['saved'], 'log': log}

    def _print_log(self, epoch, batch_idx, batch_start_time, loss):
        logger.info(f"Batch {batch_idx}, saving output ..")
