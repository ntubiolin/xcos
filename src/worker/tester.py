import os
import time
import torch
from torchvision.utils import save_image
from .worker_template import WorkerTemplate
from data_loader.base_data_loader import BaseDataLoader
from pipeline.base_pipeline import BasePipeline
from utils.global_config import global_config
from utils.logging_config import logger
from utils.verification import checkTFPN


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
            model_output = self.model(data, scenario='get_feature_and_xcos')
        return model_output, None

    def _setup_model(self):
        self.model.eval()

    def _to_log(self, epoch_stats):
        return {}

    def _init_output(self):
        """ Initialize a dictioary structure to save inferenced results. """
        for metric in self.evaluation_metrics:
            metric.clear()
        return {
            'epoch_start_time': time.time(),
            'saved': {k: [] for k in global_config.saved_keys}
        }

    def _update_output(self, epoch_output, products, write_metric=False):
        """ Update the dictionary saver: extend entries """
        self._update_all_metrics(products['data'], products['model_output'], write=write_metric)

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

        if global_config.save_while_infer and global_config.arch.type == "xCosModel":
            if global_config.arch.args.draw_qualitative_result:
                # Save results
                name = self.data_loader.name
                # print(epoch_output.keys())
                # print(epoch_output['saved'].keys())
                # print(products.keys())
                # ['flatten_feats', 'grid_feats', 'x_coses', 'attention_maps', 'grid_cos_maps', 'xcos_visualizations']
                # print(len(products['model_output']['xcos_visualizations']))
                # print(products['model_output']['xcos_visualizations'][0].shape)
                # print(epoch_output['saved'].keys())

                for i in range(len(epoch_output['saved']['xcos_visualizations'])):
                    index = epoch_output['saved']['index'][i]
                    visualization = epoch_output['saved']['xcos_visualizations'][i]
                    xcos = epoch_output['saved']['x_coses'][i]
                    is_same_label = epoch_output['saved']['is_same_labels'][i]
                    TFPN = checkTFPN(xcos, is_same_label)
                    output_path = os.path.join(self.saving_dir, f'{name}_{TFPN}_xcos_{xcos:.4f}_pair_{index:06d}.png')
                    save_image(visualization, output_path)
                    # output = {}
                    # for saved_key in epoch_output['saved'].keys():
                    #     output[saved_key] = epoch_output['saved'][saved_key][i]
                    # output_path = os.path.join(self.saving_dir, f'{name}_index{index:06d}.npz')
                    # np.savez(output_path, **output)
                    if index % 1000 == 0:
                        logger.info(f'Saving output {output_path} ...')

        return epoch_output

    def _finalize_output(self, epoch_output):
        """ Return saved inference results along with log messages """
        log = {'elasped_time (s)': time.time() - epoch_output['epoch_start_time']}
        avg_metrics = {metric.nickname: metric.finalize() for metric in self.evaluation_metrics}
        for key, value in avg_metrics.items():
            log[f"avg_{key}"] = value
        return {'saved': epoch_output['saved'], 'log': log}

    def _print_log(self, epoch, batch_idx, batch_start_time, loss):
        logger.info(f"Batch {batch_idx}, saving output ..")
