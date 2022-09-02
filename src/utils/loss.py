"""Module with custom losses."""

from torch import Tensor, nn

from src.datacls import CombinedLossOutput


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


class CombinedLoss(nn.Module):
    """Combined losses."""

    def __init__(
        self,
        first_loss: nn.Module,
        second_loss: nn.Module,
        first_coefficient: float = 1.0,
        second_coefficient: float = 1.0,
    ):
        """
        Init combinaded losses.

        Parameters
        ----------
        first_loss : first loss
        second_loss : second loss
        first_coefficient : coefficient before first loss in overall sum
        second_coefficient : coefficient before second loss in overall sum
        """
        super().__init__()
        self.coeff1 = first_coefficient
        self.coeff2 = second_coefficient
        self.loss1 = first_loss
        self.loss2 = second_loss

    def forward(self, predict: Tensor, target: Tensor) -> CombinedLossOutput:
        """
        Forward pass combination of two losses.

        Parameters
        ----------
        predict : predictions from network
        target : real targets

        Returns
        -------
        return first and second loss with it's coefficient
        """
        loss_1 = self.coeff1 * self.loss1(predict, target)
        loss_2 = self.coeff2 * self.loss2(predict, target)
        return CombinedLossOutput(loss1=loss_1, loss2=loss_2)
