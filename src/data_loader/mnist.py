import torch
from torchvision import datasets


class MnistDataset(datasets.MNIST):
    """
    Customized MNIST dataset demo
    """
    def __init__(self, data_dir, train, download, transform):
        super().__init__(data_dir, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        """ Overwrite __getitem__ to return dictionary """
        data = super().__getitem__(index)
        return {
            "data_input": data[0],
            "data_target": data[1]
        }


class MnistGANDataset(datasets.MNIST):
    """
    Customized MNIST dataset demo
    """
    def __init__(self, data_dir, train, download, transform):
        super().__init__(data_dir, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        """ Overwrite __getitem__ to return dictionary """
        data = super().__getitem__(index)
        z = torch.randn((1, 100)).view(100, 1, 1)
        return {
            "data_input": data[0],
            "data_target": data[1],
            "z": z
        }
