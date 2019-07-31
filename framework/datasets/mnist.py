import os

from .base_dataset import BaseDataset
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MnistHelper(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, index):
        return {'image': self._dataset[index][0],
                'label': self._dataset[index][1]}

    def __len__(self):
        return len(self._dataset)


class Mnist(BaseDataset):
    def _init_dataset(self, config):
        data_path = os.path.expanduser(config['data_path'])
        mnist_train_dataset = MNIST(root=data_path, train=True,
                                    download=True, transform=ToTensor())
        mnist_test_dataset = MNIST(root=data_path, train=False,
                                   download=True, transform=ToTensor())
        train_dataset = MnistHelper(mnist_train_dataset)
        test_dataset = MnistHelper(mnist_test_dataset)
        return train_dataset, test_dataset
        