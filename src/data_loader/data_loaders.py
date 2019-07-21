import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

from torchvision import transforms

from data_loader.base_data_loader import BaseDataLoader
from data_loader.mnist import MnistDataset


class MnistDataLoader(BaseDataLoader):
    """
    Customized MNIST data loader demo
    Returned data will be in dictionary
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True, name=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = MnistDataset(self.data_dir, train=training, download=True, transform=trsfm)
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
