from typing import Optional

from torch import device, nn, Tensor

from .layers import DWConv2d, DWResidualBlock


class SuperResolutionGenerator(nn.Module):
    def __init__(self, n_increase: Optional[int] = 2):
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
            DWConv2d(in_channels=64, out_channels=64),
            nn.BatchNorm2d(num_features=64),
        )
        self.residual_pixel_shuffle = self._make_conv_pixel_shuffle(n_increase)
        self.output_residual = nn.Sequential(
            DWConv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    @staticmethod
    def _make_conv_pixel_shuffle(increase: int) -> nn.Module:
        layers = []
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
        x1 = self.residual_1(x)
        x2 = self.residual_2(x1)
        x = self.residual_3(x1 + x2)
        x = self.residual_pixel_shuffle(x)
        x = self.output_residual(x)
        return x

    @property
    def device(self) -> device:
        return next(self.in_layer.parameters()).device
