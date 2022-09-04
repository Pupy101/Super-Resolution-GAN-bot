"""Module with super resolution network."""

from typing import List

from torch import Tensor, nn

from .blocks import DWConv2d, DWConv2dBNPReluBlock, DWResidualBlock, ModuleDevice


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
        self.prep_block = nn.Sequential(
            DWConv2dBNPReluBlock(in_channels=3, out_channels=8),
            DWConv2dBNPReluBlock(in_channels=8, out_channels=16),
            DWConv2dBNPReluBlock(in_channels=16, out_channels=32),
            DWConv2dBNPReluBlock(in_channels=32, out_channels=64),
        )
        self.res_block_1 = self._create_residual_block(input_shape=64)
        self.res_block_2 = self._create_residual_block(input_shape=64)
        self.res_block_3 = self._create_residual_block(input_shape=64)
        self.end_res_block = nn.Sequential(
            DWConv2d(in_channels=64, out_channels=64), nn.BatchNorm2d(num_features=64)
        )
        self.pixel_shuffle = self._make_conv_pixel_shuffle(n_increase=n_increase, input_shape=64)
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
            nn.PReLU(),
            # bottleneck like in mobilenet
            DWConv2dBNPReluBlock(in_channels=input_shape, out_channels=input_shape * 4),
            DWConv2dBNPReluBlock(in_channels=input_shape * 4, out_channels=input_shape),
            # bottleneck like in mobilenet
            DWResidualBlock(in_channels=input_shape),
            nn.PReLU(),
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
        x1 = self.prep_block(x)
        x2 = self.res_block_1(x1)
        x3 = self.res_block_2(x1 + x2)
        x4 = self.res_block_3(x1 + x2 + x3)
        x5 = self.end_res_block(x1 + x2 + x3 + x4)
        x6 = self.pixel_shuffle(x1 + x2 + x3 + x4 + x5)
        x7 = self.output_block(x6)
        return x7
