from typing import Optional

from torch import nn, Tensor

from src.datacls import CombinedLossOutput


class VGGLoss(nn.Module):
    def __init__(self, vgg: nn.Module):
        super().__init__()
        self.vgg = vgg.eval()
        self.loss = nn.MSELoss()

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        pred_logits = self.vgg(predict)
        target_logits = self.vgg(target)
        return self.loss(pred_logits, target_logits)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        first_loss: nn.Module,
        second_loss: nn.Module,
        first_coefficient: Optional[float] = 1.0,
        second_coefficient: Optional[float] = 1.0,
    ):
        super().__init__()
        self.coeff1 = first_coefficient
        self.coeff2 = second_coefficient
        self.loss1 = first_loss
        self.loss2 = second_loss

    def forward(self, predict: Tensor, target: Tensor) -> CombinedLossOutput:
        loss_1 = self.coeff1 * self.loss1(predict, target)
        loss_2 = self.coeff2 * self.loss2(predict, target)
        return CombinedLossOutput(loss1=loss_1, loss2=loss_2)
