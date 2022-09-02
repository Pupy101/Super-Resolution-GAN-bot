"""Module with layers based on DepthWise Convolutions."""

import torch
from torch import Tensor, nn


class DWConv2d(nn.Module):
    """Simple DepthWise convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """
        Init convolution layer.

        Parameters
        ----------
        in_channels : count input channels
        out_channels : count output channels
        kernel_size : kernel size of convolution
        stride : stride of convolution
        padding : padding of convolution
        """
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
        """
        Layer forward pass.

        Parameters
        ----------
        x : input tensor

        Returns
        -------
        output transformed tensor
        """
        return self.conv(x)

    @property
    def device(self) -> torch.device:
        """
        Get layer device.

        Returns
        -------
        network device
        """
        return next(self.parameters()).device


class DWResidualBlock(nn.Module):
    """Residual block with sequence of DepthWise + BN + PReLU + DepthWise + BN."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """
        Init residual block.

        Parameters
        ----------
        in_channels : count input channels
        kernel_size : kernel size of convolution
        stride : stride of convolution
        padding : padding of convolution
        """
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
        """
        Residual block forward pass.

        Parameters
        ----------
        x : input tensor

        Returns
        -------
        output transformed tensor
        """
        return self.block(in_features) + in_features

    @property
    def device(self) -> torch.device:
        """
        Get block device.

        Returns
        -------
        network device
        """
        return next(self.parameters()).device
