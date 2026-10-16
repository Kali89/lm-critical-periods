"""ResNet-18, CIFAR variant (3x3 stem, no maxpool), NHWC, BatchNorm, float32."""
import mlx.core as mx
import mlx.nn as nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm(cout))

    def __call__(self, x):
        y = nn.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return nn.relu(y + (x if self.short is None else self.short(x)))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(64)
        blocks, cin = [], 64
        for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            blocks += [Block(cin, cout, stride), Block(cout, cout, 1)]
            cin = cout
        self.blocks = blocks
        self.fc = nn.Linear(512, num_classes)

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        for b in self.blocks:
            x = b(x)
        return self.fc(mx.mean(x, axis=(1, 2)))
