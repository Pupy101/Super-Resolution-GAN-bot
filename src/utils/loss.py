"""Module with custom losses."""

from torch import Tensor, nn


class VGGLoss(nn.Module):
    """VGGLoss."""

    def __init__(self, vgg: nn.Module):
        """
        Init loss.

        Parameters
        ----------
        vgg : vgg network
        """
        super().__init__()
        self.vgg = vgg.eval()
        self.loss = nn.MSELoss()

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass VGGLoss.

        Parameters
        ----------
        predict : predictions from network
        target : real targets

        Returns
        -------
        VGGLoss between predict and target
        """
        pred_logits = self.vgg(predict)
        target_logits = self.vgg(target)
        return self.loss(pred_logits, target_logits)
