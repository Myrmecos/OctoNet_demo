"""ResNet-18 with a configurable number of input channels.

The layout follows the torchvision ResNet-18: a 7x7 stride-2 stem, a max pool,
and four stages of two basic blocks. Adaptive average pooling accepts the small
time-frequency and spatial maps produced from each OctoNet modality.
"""

import torch
from torch import nn
from torch.nn import functional as F


class BatchNorm2d(nn.BatchNorm2d):
    """BatchNorm that stays defined when a 1x1 map is paired with batch size 1.

    Small modality tensors reach 1x1 before the last stage. A single-example
    batch then has one value per channel, which BatchNorm cannot normalize.
    In that case the layer uses its running statistics.
    """

    def forward(self, x):
        if self.training and x.shape[0] * x.shape[2] * x.shape[3] < 2:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps
            )
        return super().forward(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._stage(64, 64, n_blocks=2, stride=1)
        self.layer2 = self._stage(64, 128, n_blocks=2, stride=2)
        self.layer3 = self._stage(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._stage(256, 512, n_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.reset_parameters()

    def _stage(self, in_channels, out_channels, n_blocks, stride):
        blocks = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))
