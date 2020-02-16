import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

from torchvision import transforms

from .base_data_loader import BaseDataLoader
from .mnist import MnistDataset
from .mnist_result import MnistResultDataset
from .face_datasets import SiameseImageFolder, InsightFaceBinaryImg


class FaceDataLoader(BaseDataLoader):
    """
    Customized MNIST data loader demo
    Returned data will be in dictionary
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=1, name=None,
                 norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
        trsfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
        self.data_dir = data_dir
        self.dataset = SiameseImageFolder(data_dir, trsfm)
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class FaceBinDataLoader(BaseDataLoader):
    """
    Customized Face data loader that load val data from bin files
    Returned data will be in dictionary
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=1, name="LFW",
                 norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
        self.data_dir = data_dir
        self.dataset = InsightFaceBinaryImg(data_dir, name, trsfm)
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MnistDataLoader(BaseDataLoader):
    """
    Customized MNIST data loader demo
    Returned data will be in dictionary
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True, name=None,
                 img_size=28, norm_mean=(0.1307,), norm_std=(0.3081,)):
        trsfm = transforms.Compose([
            transforms.Scale(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
        self.data_dir = data_dir
        self.dataset = MnistDataset(self.data_dir, train=training, download=True, transform=trsfm)
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MnistResultDataLoader(BaseDataLoader):
    """
    Customized MNIST result data loader demo
    Returned data will be in dictionary
    """
    def __init__(self, dataset_args, batch_size, num_workers=1, training=True, name=None):
        self.dataset = MnistResultDataset(**dataset_args)
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(self.dataset, batch_size, False, 0, num_workers)
