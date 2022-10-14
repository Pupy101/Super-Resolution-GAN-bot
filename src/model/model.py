"""Module with super resolution network."""

from typing import List, Tuple

from torch import Tensor, nn

from .blocks import DWConv2d, DWConv2dBNPReluBlock, DWResidualBlock, ModuleDevice


class SuperResolutionGenerator(ModuleDevice):
    """Super resolution network."""

    def __init__(
        self,
        n_increase: int = 2,
        count_residual_blocks: int = 30,
        in_shapes: Tuple[int, ...] = (8, 16, 32),
        inner_shape: int = 64,
        out_shapes: Tuple = (32, 16, 8),
    ):
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
        # create in blocks
        in_blocks: List[nn.Module] = [
            DWConv2dBNPReluBlock(in_channels=3, out_channels=in_shapes[0])
        ]
        for i in range(len(in_shapes) - 1):
            in_blocks.append(
                DWConv2dBNPReluBlock(
                    in_channels=in_shapes[i], out_channels=in_shapes[i + 1]
                )
            )
        in_blocks.append(
            DWConv2dBNPReluBlock(in_channels=in_shapes[-1], out_channels=inner_shape)
        )
        self.in_blocks = nn.Sequential(*in_blocks)
        # create residual blocks
        self.res_blocks = nn.ModuleList(
            [
                DWResidualBlock(in_channels=inner_shape)
                for _ in range(count_residual_blocks)
            ]
        )
        # create pixel shuffle for upscale image
        self.pixel_shuffle = self._make_conv_pixel_shuffle(
            n_increase=n_increase, input_shape=inner_shape
        )
        # create output blocks
        out_blocks: List[nn.Module] = [
            DWConv2dBNPReluBlock(in_channels=inner_shape, out_channels=out_shapes[0])
        ]
        for i in range(len(out_shapes) - 1):
            out_blocks.append(
                DWConv2dBNPReluBlock(
                    in_channels=out_shapes[i], out_channels=out_shapes[i + 1]
                )
            )
        out_blocks.append(DWConv2d(in_channels=out_shapes[-1], out_channels=3))
        self.out_blocks = nn.Sequential(*out_blocks)

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
        for _ in range(n_increase // 2):
            layers.extend(
                [
                    DWConv2d(in_channels=input_shape, out_channels=input_shape * 4),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.PReLU(),
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
        x = self.in_blocks(x)
        x_res = x
        for res_block in self.res_blocks:
            x_res = res_block(x_res)
        pixel_shuffle_output = self.pixel_shuffle(x_res + x)
        return self.out_blocks(pixel_shuffle_output)
