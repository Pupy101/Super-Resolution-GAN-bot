"""Module with block based on DepthWise Convolutions."""

import torch
from torch import Tensor, nn


class ModuleDevice(nn.Module):
    """Base class for get property with device of network."""

    @property
    def device(self) -> torch.device:
        """
        Get module device.

        Returns
        -------
        network device
        """
        return next(self.parameters()).device


class DWConv2d(ModuleDevice):
    """DepthWise convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        """
        Init DepthWise convolution.

        Parameters
        ----------
        in_channels : count input channels
        out_channels : count output channels
        kernel_size : kernel size of convolution
        stride : stride of convolution
        padding : padding of convolution
        """
        super().__init__()
        self.dw_conv = nn.Sequential(
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
        DepthWise convolution forward pass.

        Parameters
        ----------
        x : input tensor

        Returns
        -------
        output transformed tensor
        """
        return self.dw_conv(x)


class DWConv2dBNPReluBlock(ModuleDevice):
    """Block with sequence of DepthWise convolution + BN + PReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """
        Init block with DepthWise convolution, BN, PReLU.

        Parameters
        ----------
        in_channels : count input channels
        kernel_size : kernel size of convolution
        stride : stride of convolution
        padding : padding of convolution
        """
        super().__init__()
        self.dw_conv_bn_prelu = nn.Sequential(
            DWConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Block forward pass.

        Parameters
        ----------
        x : input tensor

        Returns
        -------
        output transformed tensor
        """
        return self.dw_conv_bn_prelu(x)


class DWResidualBlock(ModuleDevice):
    """Residual block with sequence of DepthWise convolution + BN + PReLU + DepthWise convolution + BN."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """
        Init residual block (DepthWise convolution + BN + PReLU + DepthWise convolution + BN).

        Parameters
        ----------
        in_channels : count input channels
        kernel_size : kernel size of convolution
        stride : stride of convolution
        padding : padding of convolution
        """
        super().__init__()
        self.dw_res_block = nn.Sequential(
            DWConv2dBNPReluBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            DWConv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Block forward pass.

        Parameters
        ----------
        x : input tensor

        Returns
        -------
        output transformed tensor
        """
        return self.dw_res_block(x) + x
