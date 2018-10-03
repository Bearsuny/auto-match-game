import torch
import torch.nn as nn


class RewardNet(nn.Module):
    def __init__(self):
        super(RewardNet, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 1),
            nn.ReLU(),
            nn.MaxPool2d(1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )
        self.result = nn.Sequential(
            nn.Linear(256, 8)
        )
        self._features = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )
        self._classifier = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3
        )

    def forward(self, x):
        output = self._features(x)
        output = output.view(output.size(0), -1)
        output = self._classifier(output)
        return self.result(output)

    def _initialize_weights(self):
        for model_item in self.modules():
            if isinstance(model_item, nn.Conv2d):
                model_item.weight.data.zero_()
                model_item.weight.data.normal_(0.0, 0.1)
                if model_item.bias is not None:
                    model_item.bias.data.zero_()
