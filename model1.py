import torch.nn as nn
import torch.nn.functional as F
import torch

dropout_value = 0.1


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # rin: 1, rout: 3, in_size: 28, out_size:28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
        )

        # rin: 3, rout: 5, in_size: 28, out_size:28
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 10, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
        )

        # rin: 14, rout: 16, in_size: 7, out_size:7
        self.pool1 = nn.MaxPool2d(2, 2)
        self.ant1 = nn.Conv2d(10, 8, 1)

        # rin: 6, rout: 10, in_size: 14, out_size:14
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
        )

        # rin: 10, rout: 14, in_size: 14, out_size:14
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 10, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
        )

        # rin: 14, rout: 16, in_size: 7, out_size:7
        self.pool2 = nn.MaxPool2d(2, 2)
        self.ant2 = nn.Conv2d(10, 8, 1)

        # rin: 16, rout: 24, in_size: 7, out_size:5
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
        )

        # rin: 24, rout: 32, in_size: 5, out_size:3
        self.conv6 = nn.Sequential(
            nn.Conv2d(8, 10, 3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )
        self.linear = nn.Linear(10 * 5 * 5, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.ant1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.ant2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 10 * 5 * 5)
        x = self.linear(x)
        return F.log_softmax(x)
