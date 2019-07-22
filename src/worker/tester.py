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
