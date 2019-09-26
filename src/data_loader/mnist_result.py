import numpy as np
from torch.utils.data import Dataset


class MnistResultDataset(Dataset):
    """
    Customized MNIST result dataset demo
    """
    def __init__(self, result_filename, key='model_output'):
        self.key = key
        self.results = self._load_data(result_filename, key)

    def _load_data(self, result_filename, key):
        return np.load(result_filename)[key]

    def __getitem__(self, index):
        """ Overwrite __getitem__ to return dictionary """
        result = self.results[index]
        return {
            "index": index,
            self.key: result
        }

    def __len__(self):
        return len(self.results)
