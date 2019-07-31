import torch
import torch.nn as nn
import torch.nn.functional as func
from .base_model import BaseModel, Mode


class SimpleClassifierModule(nn.Module):
    def __init__(self):
        super(SimpleClassifierModule, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(7 * 7 * 64, 1024)
        self.linear2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(1)
    
    def forward(self, inputs):
        x = func.relu(self.conv1(inputs))
        x = self.pool1(x)
        x = func.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = func.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def class_prob(self, x):
        return self.softmax(x)

    def pred_class(self, x):
        return torch.argmax(x, 1)


class SimpleClassifier(BaseModel):
    required_config_keys = []

    def __init__(self, dataset, config, device):
        super(SimpleClassifier, self).__init__(dataset, config, device)
        self.loss = nn.CrossEntropyLoss()

    def _model(self, config):
        return SimpleClassifierModule()

    def _forward(self, inputs, mode, config):
        x = self._net.forward(inputs['image'])

        if mode == Mode.TRAIN:
            return {'logits': x}
        else:
            return {'logits': x, 'prob': self._net.class_prob(x),
                    'pred': self._net.pred_class(x)}

    def _loss(self, outputs, inputs, config):
        loss = self.loss(outputs['logits'], inputs['label'])
        return loss

    def _metrics(self, outputs, inputs, config):
        correct_count = torch.eq(outputs['pred'], inputs['label']).float()
        metrics = {'accuracy': torch.mean(correct_count)}
        return metrics

    def initialize_weights(self):
        def init_weights(m):
            if (type(m) == nn.Conv2d) or (type(m) == nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)

        self._net.apply(init_weights)
