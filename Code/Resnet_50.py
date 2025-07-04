import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    # Resnet18 Resnet34
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        out = self.residual_function(x) + self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out


class Bottleneck(nn.Module):
    # Resnet50+ layers
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x):
        out = self.residual_function(x) + self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out


class ResNet(nn.Module):

    def __init__(self, in_channels, block, num_block, num_classes=21):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # one layer may contain more than one residual block
        '''
        :param block: block type, basic block
        :param out_channels: output depth channel number of this layer
        :param num_blocks: how many blocks per layer
        :param stride: stride of the first block of this layer
        :return: a resnet layer
        '''
        # first block could be 1 or 2, other would always be 1

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.conv2_x(x)
        out.append(x)
        x = self.conv3_x(x)
        out.append(x)
        x = self.conv4_x(x)
        out.append(x)
        x = self.conv5_x(x)
        out.append(x)
        # out = self.avg_pool(out)
        return out


def ResNet18(in_channel=3, num_classes=21):
    return ResNet(in_channel, BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(in_channel=3, num_classes=21):
    return ResNet(in_channel, BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(in_channel=3, num_classes=21):
    return ResNet(in_channel, Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(in_channel=3, num_classes=21):
    return ResNet(in_channel, Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(in_channel=3, num_classes=21):
    return ResNet(in_channel, Bottleneck, [3, 8, 36, 3], num_classes)
