from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.activ = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, in_features):
        x = self.conv1(in_features)
        x = self.bn1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + in_features
