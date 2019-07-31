import os
import logging
import numpy as np
import torch
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


class Mode:
    TRAIN = 'train'
    TEST = 'test'
    PRED = 'pred'


class BaseModel(metaclass=ABCMeta):
    """Base model class.

    Arguments:
        dataset: A BaseDataset object.
        config: A dictionary containing the configuration parameters.
                The Entry `learning_rate` is required.

    Models should inherit from this class and implement the following methods:
        `_model`, `_loss`, `_metrics` and `initialize_weights`.

    Additionally, the static attribute required_config_keys is a list
    containing the required config entries.
    """
    required_baseconfig = ['learning_rate']

    @abstractmethod
    def _model(self, config):
        """ Implements the model.

        Arguments:
            config: A configuration dictionary.

        Returns:
            A torch.nn.Module that implements the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _forward(self, inputs, mode, config):
        """ Calls the model on some input.
        This method is called three times: for training, testing and prediction
        (see the `mode` argument) and can return different tensors depending
        on the mode. It only supports NCHW format for now.

        Arguments:
            inputs: A dictionary of input features, where the keys are their
                names (e.g. `"image"`) and the values of type `torch.Tensor`.
            mode: An attribute of the `Mode` class, either `Mode.TRAIN`,
                  `Mode.TEST` or `Mode.PRED`.
            config: A configuration dictionary.

        Returns:
            A dictionary of outputs, where the keys are their names
            (e.g. `"logits"`) and the values are the corresponding Tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def _loss(self, outputs, inputs, config):
        """ Implements the training loss.
        This method is called on the outputs of the `_model` method
        in training mode.

        Arguments:
            outputs: A dictionary, as returned by `_model` called with
                     `mode=Mode.TRAIN`.
            inputs: A dictionary of input features (same as for `_model`).
            config: A configuration dictionary.

        Returns:
            A Tensor corresponding to the loss to be minimized during training.
        """
        raise NotImplementedError

    @abstractmethod
    def _metrics(self, outputs, inputs, config):
        """ Implements the evaluation metrics.
        This method is called on the outputs of the `_model` method
        in test mode.

        Arguments:
            outputs: A dictionary, as returned by `_model` called with
                     `mode=Mode.EVAL`.
            inputs: A dictionary of input features (same as for `_model`).
            config: A configuration dictionary.

        Returns:
            A dictionary of metrics, where the keys are their names
            (e.g. "`accuracy`") and the values are the corresponding Tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_weights(self):
        """ Initialize all the weights in the network. """
        return NotImplementedError

    def __init__(self, dataset, config, device):
        self._dataset = dataset
        self._config = config
        required = self.required_baseconfig + getattr(
            self, 'required_config_keys', [])
        for r in required:
            assert r in self._config, 'Required configuration entry: \'{}\''.format(r)
        self._net = self._model(config).to(device)
        self._solver = optim.Adam(self._net.parameters(),
                                  lr=self._config['learning_rate'])
        self._it = 0
        self._epoch = 0

    def _to_dict(self, d, device):
        """ Send all the values of a dict of Tensor to a specific device. """
        return {k: v.to(device) for k, v in d.items()}

    def train(self, n_iter, n_epoch, exper_dir, validation_interval=100,
              save_interval=500, resume_training='', device='cpu'):
        """ Train the model.

        Arguments:
            n_iter: max number of iterations.
            n_epoch: max number of epochs.
            exper_dir: folder containing the outputs of the training.
            validation_interval: evaluate the model every
                                 'validation_interval' iter.
            save_interval: save the model every 'save_interval' iter.
            resume_training: path to the checkpoint file when training is
                             resumed from a previous training.
            device: device on which to perform the operations.
        """
        if resume_training == '':
            logging.info('Initializing new weights.')
            self.initialize_weights()
            self._it = 0
            self._epoch = 0
        else:
            logging.info('Loading weights from ' + resume_training)
            self.load(resume_training)
        self._net.train()

        runs_dir = os.path.join(exper_dir, 'runs')
        writer = SummaryWriter(runs_dir)
        train_data_loader = self._dataset.get_train_dataloader()
        val_data_loader = self._dataset.get_validation_dataloader()

        train_loss = []
        logging.info('Start training')
        while self._epoch < n_epoch:
            for x in train_data_loader:
                if self._it > n_iter:
                    break
                self._solver.zero_grad()

                # Optimize one batch
                inputs = self._to_dict(x, device)
                outputs = self._forward(inputs, Mode.TRAIN, self._config)
                loss = self._loss(outputs, inputs, self._config)
                train_loss.append(loss.item())
                loss.backward()
                self._solver.step()

                # Validation
                if self._it % validation_interval == 0:
                    validation_loss = []
                    metrics = {}
                    for x in val_data_loader:
                        inputs = self._to_dict(x, device)
                        outputs = self._forward(inputs, Mode.TEST,
                                                self._config)
                        loss = self._loss(outputs, inputs, self._config)
                        validation_loss.append(loss.item())
                        metric = self._metrics(outputs, inputs, self._config)
                        metrics = {k: [metric[k].item()] if k not in metrics
                                   else metrics[k] + [metric[k].item()]
                                   for k in metric}
                    total_train_loss = np.mean(train_loss)
                    total_validation_loss = np.mean(validation_loss)
                    total_metrics = {k: np.mean(v) for k, v in metrics.items()}

                    # Print the metrics
                    logging.info(
                        'Iter {:4d}: train loss {:.4f}, validation loss {:.4f}'.format(
                            self._it, total_train_loss, total_validation_loss)
                        + ''.join([', {} {:.4f}'.format(m, total_metrics[m])
                                   for m in total_metrics]))

                    # Save them a summary files
                    writer.add_scalar('train_loss', total_train_loss, self._it)
                    writer.add_scalar('validation_loss', total_validation_loss,
                                      self._it)
                    for m in total_metrics:
                        writer.add_scalar(m, total_metrics[m], self._it)
                
                if self._it % save_interval == 0:
                    self.save(exper_dir)

                self._it += 1
            self._epoch += 1
            if self._it > n_iter:
                break
        writer.close()
        logging.info('Training finished.')
    
    def test(self, exper_dir, checkpoint_path, device='cpu'):
        """ Test the model on a test dataset.

        Arguments:
            exper_dir: folder containing the outputs of the training.
            checkpoint_path: path to the checkpoint.
            device: device on which to perform the operations.
        """
        # Load the weights
        logging.info('Loading weights from ' + checkpoint_path)
        self.load(checkpoint_path)
        self._net.eval()
        test_data_loader = self._dataset.get_test_dataloader()

        # Run the evaluation metrics on the test dataset
        logging.info('Start evaluation.')
        metrics = {}
        for x in test_data_loader:
            inputs = self._to_dict(x, device)
            outputs = self._forward(inputs, Mode.TEST,
                                    self._config)
            metric = self._metrics(outputs, inputs, self._config)
            metrics = {k: [metric[k].item()] if k not in metrics
                        else metrics[k] + [metric[k].item()]
                        for k in metric}
        total_metrics = {k: np.mean(v) for k, v in metrics.items()}
        logging.info('Test metrics: '
                     + ''.join(['{} {:.4f};'.format(m, total_metrics[m])
                                for m in total_metrics]))


    def save(self, exper_dir):
        """ Save the current training in a .pth file. """
        save_file = os.path.join(
            exper_dir, 'checkpoints/checkpoint_' + str(self._it) + '.pth')
        torch.save({'iter': self._it,
                    'model_state_dict': self._net.state_dict(),
                    'optimizer_state_dict': self._solver.state_dict()},
                   save_file)

    def load(self, checkpoint_path):
        """ Load a model stored on disk. """
        checkpoint = torch.load(checkpoint_path)
        self._net.load_state_dict(checkpoint['model_state_dict'])
        self._solver.load_state_dict(checkpoint['optimizer_state_dict'])
        self._it = checkpoint['iter']
        self._epoch = (
            self._it * self._dataset.get_train_dataloader().batch_size
            // len(self._dataset._train_dataset))