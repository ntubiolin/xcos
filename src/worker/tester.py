import torch

from .worker_template import WorkerTemplate
from pipeline.base_pipeline import BasePipeline


class Tester(WorkerTemplate):
    """
    Tester class

    Note:
        Inherited from WorkerTemplate.
    """
    def __init__(self, pipeline: BasePipeline, *args):
        super().__init__(pipeline, *args)
        # Some shared attributes are trainer exclusive and therefore is initialized here
        for attr_name in ['saved_keys']:
            setattr(self, attr_name, getattr(pipeline, attr_name))

    @property
    def enable_grad(self):
        return False

    def _run_and_optimize_model(self, data):
        model_output = self.model(data)
        return model_output, None, None

    def _setup_model(self):
        self.model.eval()

    def _to_log(self, epoch_stats):
        return {}

    def _stats_init(self):
        """ Initialize a dictioary structure to save inferenced results. """
        return {k: [] for k in self.saved_keys}

    def _stats_update(self, stats, products):
        """ Update the dictionary saver. """
        data, model_output = products['data'], products['model_output']

        def fetch_from_dict(dictionary):
            for key in dictionary.keys():
                if key not in self.saved_keys:
                    continue
                value = dictionary[key]
                saved_value = value.cpu().numpy() if torch.is_tensor(value) else value
                stats[key].extend([v for v in saved_value])

        fetch_from_dict(data)
        fetch_from_dict(model_output)
        return stats

    def _stats_finalize(self, stats):
        """ Just return the dictionary saver. """
        return stats

    def _finalize_output(self, epoch_stats):
        """ Just return the dictionary saver. """
        return epoch_stats

#     def inference(self, data_loader, saved_keys=['verb_logits', 'noun_logits', 'uid']):
#         self.model.eval()
#         logger.info(f'Inferencing with following keys to save: {saved_keys}')
#         logger.info(f'Number of examples is around {data_loader.batch_size * len(data_loader)}')
#         saved_results = {k: [] for k in saved_keys}

#         with torch.no_grad():
#             for batch_idx, data_input in enumerate(data_loader):
#                 self.writer.set_step(batch_idx, 'inference')
#                 for key in data_input.keys():
#                     value = data_input[key]
#                     if key in saved_keys:
#                         saved_value = value.numpy() if torch.is_tensor(value) else value
#                         saved_results[key].extend([v for v in saved_value])
#                     data_input[key] = value.to(self.device) if torch.is_tensor(value) else value

#                 model_output = self.model(data_input)
#                 for key in model_output.keys():
#                     if key in saved_keys:
#                         saved_results[key].extend([v for v in model_output[key].cpu().numpy()])

#                 if batch_idx == 0:
#                     non_exists_keys = [key for key in saved_keys if len(saved_results[key]) == 0]
#                     if len(non_exists_keys) > 0:
#                         logger.warning(f'Keys {non_exists_keys} not exists')
#                 if batch_idx % 10 == 0:
#                     logger.info(f'Entry {batch_idx * data_loader.batch_size} done.')

#         return saved_results
