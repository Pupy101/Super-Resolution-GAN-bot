from typing import Optional

from torch import nn, Tensor


class DWConv2d(nn.Module):
    """
    Model for x4 resolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[int] = 3,
        stride: Optional[int] = 1,
        padding: Optional[int] = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class DWResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Optional[int] = 3,
        stride: Optional[int] = 1,
        padding: Optional[int] = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            DWConv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.PReLU(),
            DWConv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=in_channels),
        )

    def forward(self, in_features: Tensor) -> Tensor:
        return self.block(in_features) + in_features
