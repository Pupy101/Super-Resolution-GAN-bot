"""Module with super resolution network."""

from typing import List

import torch
from torch import Tensor, nn

from .layers import DWConv2d, DWResidualBlock, ModuleDevice


class SuperResolutionGenerator(ModuleDevice):
    """Super resolution network."""

    def __init__(self, n_increase: int = 2):
        """
        Init network.

        Parameters
        ----------
        n_increase : increasing resolution multiplier
        """
        self.n_increase = n_increase
        assert not n_increase % 2, "Increase must be multiple of 2"
        super().__init__()
        self.residual_1 = nn.Sequential(
            DWConv2d(in_channels=3, out_channels=8),
            nn.PReLU(),
            DWConv2d(in_channels=8, out_channels=16),
            nn.PReLU(),
            DWConv2d(in_channels=16, out_channels=32),
            nn.PReLU(),
            DWConv2d(in_channels=32, out_channels=64),
            nn.PReLU(),
        )
        self.residual_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWResidualBlock(in_channels=64),
        )
        self.residual_3 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWResidualBlock(in_channels=64),
            nn.BatchNorm2d(num_features=64),
        )
        self.residual_pixel_shuffle = self._make_conv_pixel_shuffle(n_increase)
        self.output_residual = nn.Sequential(
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWResidualBlock(in_channels=64),
            nn.PReLU(),
            DWConv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    @staticmethod
    def _make_conv_pixel_shuffle(increase: int) -> nn.Module:
        """
        Create incrasing layers.

        Parameters
        ----------
        increase : increasing resolution multiplier

        Returns
        -------
        Increasing layers
        """
        layers: List[nn.Module] = []
        layers.append(nn.BatchNorm2d(64))
        for _ in range(increase // 2):
            layers.extend(
                [
                    DWConv2d(in_channels=64, out_channels=256),
                    nn.PixelShuffle(2),
                    nn.PReLU(),
                    DWResidualBlock(in_channels=64),
                    nn.PReLU(),
                    DWResidualBlock(in_channels=64),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Network forward pass.

        Parameters
        ----------
        x : input tensor

        Returns
        -------
        output tensor
        """
        x1 = self.residual_1(x)
        x2 = self.residual_2(x1)
        x3 = self.residual_3(x1 + x2)
        x4 = self.residual_pixel_shuffle(x1 + x2 + x3)
        x5 = self.output_residual(x4)
        return x5
