import numpy as np
from torch.utils.data import Dataset


class MnistResultDataset(Dataset):
    """
    Customized MNIST result dataset demo
    """
    def __init__(self, result_filename):
        self.results = self._load_data(result_filename)

    def _load_data(self, result_filename):
        return np.load(result_filename)['model_output']

    def __getitem__(self, index):
        """ Overwrite __getitem__ to return dictionary """
        result = self.results[index]
        return {
            "model_output": result
        }

    def __len__(self):
        return len(self.results)
