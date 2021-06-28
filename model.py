import torch

from torch import nn
from utils.model_blocks import ResidualBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.residual = nn.Sequential(
            ResidualBlock(channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.last_residual = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64)
        )
        self.conv_pixelx2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.last_layers = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        res = self.in_layer(x)
        out_res = self.residual(res)
        out = self.last_residual(out_res + res)
        out = self.conv_pixelx2(out)
        out = self.last_layers(out)
        return out
