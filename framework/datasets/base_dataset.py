import numpy as np
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataset(metaclass=ABCMeta):
    """Base dataset class.

    Arguments:
        config: A dictionary containing the configuration parameters.

    Datasets should inherit from this class and implement the method
    `_init_dataset`.

    Additionally, the static attribute required_config_keys contains a
    list with the required config entries.
    """
    required_baseconfig = ['batch_size', 'test_batch_size']

    @abstractmethod
    def _init_dataset(self, config):
        """Prepare the dataset for reading.

        This method should configure the dataset for later fetching, such as
        downloading the data if it is not stored locally, or reading the list
        of data files from disk. Ideally, especially in the case of large
        images, this method shoudl NOT read all the dataset into memory,
        but rather prepare for faster subsequent fetching.

        Arguments:
            config: A configuration dictionary.
        Returns:
            A tuple of torch.utils.data.Dataset objects yielding a training
             and test dataset.
        """
        raise NotImplementedError

    def __init__(self, config):
        self._config = config
        required = self.required_baseconfig + getattr(
            self, 'required_config_keys', [])
        for r in required:
            assert r in self._config, 'Required configuration entry: \'{}\''.format(r)
        self._train_dataset, self._test_dataset = self._init_dataset(config)

        # Split the training set in train and validation subsets
        validation_split = self._config.get('validation_split', 0.1)
        seed = self._config.get('seed', 0)
        train_size = len(self._train_dataset)
        indices = list(range(train_size))
        split = int(np.floor(validation_split * train_size))
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self._train_dataloader = DataLoader(
            self._train_dataset, batch_size=config['batch_size'],
            sampler=train_sampler, num_workers=4)
        self._validation_dataloader = DataLoader(
            self._train_dataset, batch_size=config['test_batch_size'],
            sampler=valid_sampler, num_workers=4)
        self._test_dataloader = DataLoader(
            self._test_dataset, batch_size=config['test_batch_size'],
            shuffle=False, num_workers=4)

    def get_train_dataset(self):
        return self._train_dataset
    
    def get_train_dataloader(self):
        return self._train_dataloader
    
    def get_validation_dataloader(self):
        return self._validation_dataloader

    def get_test_dataset(self):
        return self._test_dataset
    
    def get_test_dataloader(self):
        return self._test_dataloader