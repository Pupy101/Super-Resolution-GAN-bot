"""Module with super resolution network."""

from functools import reduce
from operator import add
from typing import List

from torch import Tensor, nn

from .blocks import DWConv2d, DWConv2dBNPReluBlock, DWResidualBlock, ModuleDevice


class SuperResolutionGenerator(ModuleDevice):
    """Super resolution network."""

    def __init__(self, n_increase: int = 2, count_residual_blocks: int = 4):
        """
        Init network.

        Parameters
        ----------
        n_increase : increasing resolution multiplier
        count_residual_blocks : count residual blocks DWResidualBlock in net
        """
        self.n_increase = n_increase
        assert not n_increase % 2, "Increase must be multiple of 2"
        super().__init__()
        self.prep_block = nn.Sequential(
            DWConv2dBNPReluBlock(in_channels=3, out_channels=8),
            DWConv2dBNPReluBlock(in_channels=8, out_channels=16),
            DWConv2dBNPReluBlock(in_channels=16, out_channels=32),
            DWConv2dBNPReluBlock(in_channels=32, out_channels=64),
        )
        self.res_blocks = nn.ModuleList(
            [
                self._create_residual_block(input_shape=64)
                for _ in range(count_residual_blocks)
            ]
        )
        self.pixel_shuffle = self._make_conv_pixel_shuffle(
            n_increase=n_increase, input_shape=64
        )
        self.output_block = nn.Sequential(
            DWConv2dBNPReluBlock(in_channels=64, out_channels=32),
            DWConv2dBNPReluBlock(in_channels=32, out_channels=16),
            DWConv2dBNPReluBlock(in_channels=16, out_channels=8),
            DWConv2d(in_channels=8, out_channels=3),
            nn.Tanh(),
        )

    @staticmethod
    def _create_residual_block(input_shape: int) -> nn.Module:
        return nn.Sequential(
            DWResidualBlock(in_channels=input_shape),
            # bottleneck like in mobilenet
            DWConv2dBNPReluBlock(in_channels=input_shape, out_channels=input_shape * 4),
            DWConv2dBNPReluBlock(in_channels=input_shape * 4, out_channels=input_shape),
            # bottleneck like in mobilenet
            DWResidualBlock(in_channels=input_shape),
            DWResidualBlock(in_channels=input_shape),
        )

    @staticmethod
    def _make_conv_pixel_shuffle(n_increase: int, input_shape: int) -> nn.Module:
        """
        Create incrasing layers.

        Parameters
        ----------
        n_increase : increasing resolution multiplier

        Returns
        -------
        Increasing layers
        """
        layers: List[nn.Module] = []
        layers.append(nn.BatchNorm2d(input_shape))
        for _ in range(n_increase // 2):
            layers.extend(
                [
                    DWConv2d(in_channels=input_shape, out_channels=input_shape * 4),
                    nn.PixelShuffle(2),
                    nn.PReLU(),
                    DWResidualBlock(in_channels=input_shape),
                    nn.PReLU(),
                    DWResidualBlock(in_channels=input_shape),
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
        x = self.prep_block(x)
        res_blocks_inputs = [x]
        for res_block in self.res_blocks:
            block_input = reduce(add, res_blocks_inputs)
            output = res_block(block_input)
            res_blocks_inputs.append(output)
        pixel_shuffle_input = reduce(add, res_blocks_inputs)
        pixel_shuffle_output = self.pixel_shuffle(pixel_shuffle_input)
        return self.output_block(pixel_shuffle_output)
