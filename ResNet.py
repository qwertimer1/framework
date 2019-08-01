"""
Neural Net Architecture models
"""

from torch import nn


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    """Basic 3x3 convolutional structure

    Arguments:
        in_channels: Number of Input channels
        out_channels: Number of output channels

    Keyword Arguments:
        stride: Convolutional stride

    Returns:
        2D convolution output
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block


class ResidualBlock(nn.Module):
    """Residual Blocks - https://arxiv.org/pdf/1512.03385.pdf

    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        """Shape of residual block

        Arguments:
            x: Input

        Returns:
            out: Output
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet


class ResNet(nn.Module):
    """ResNet Model - https://arxiv.org/pdf/1512.03385.pdf

    """

    def __init__(self, block, layers, in_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.conv = conv3x3(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)

        self.avg_pool = nn.AvgPool2d(32)
        self.fc = nn.Linear(1024, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        """Build Residual block layers

        Args:
            block: Residual block
            out_channels: -- [description]
            blocks: Number of Residual blocks per layer

        Keyword Arguments:
            stride: Convolutional stride

        """
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """forward pass of the Resnet model

        Args:
            x: Input
        Returns:
            output
        """
        out = self.conv(x)

        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)
      
        out = self.avg_pool(out)
    
        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out
