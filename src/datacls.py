from dataclasses import dataclass
from typing import Any, Dict, List

from torch import nn, optim, Tensor, device
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


@dataclass  # data.augmentation
class Augmentation:
    mean: List[float]
    std: List[float]
    main: T.Compose
    large: T.Compose
    small: T.Compose


@dataclass
class InferenceAugmentation:
    mean: List[float]
    std: List[float]
    main: T.Compose
    resize: T.Resize


@dataclass  # train.train
class GANParameters:
    generator: Any
    discriminator: Any


@dataclass
class TorchModuleSubclass(GANParameters):
    def state_dict(self) -> Dict[str, Dict[str, Tensor]]:
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def load_state_dict(self, weight: Dict[str, Dict[str, Tensor]]) -> None:
        assert "generator" in weight, "Not right weights. Not founded generator"
        assert "discriminator" in weight, "Not right weights. Not founded discriminator"
        self.generator.load_state_dict(weight["generator"])
        self.discriminator.load_state_dict(weight["discriminator"])


@dataclass
class Model(TorchModuleSubclass):
    generator: nn.Module
    discriminator: nn.Module


@dataclass
class Optimizer(TorchModuleSubclass):
    generator: optim.Optimizer
    discriminator: optim.Optimizer


@dataclass
class CriterionGenerator:
    mse_vgg: nn.Module
    bce: nn.Module


@dataclass
class Criterion(GANParameters):
    generator: CriterionGenerator
    discriminator: nn.Module


@dataclass
class Datasets:
    train: Dataset
    valid: Dataset


@dataclass
class Dataloaders(Datasets):
    train: DataLoader
    valid: DataLoader


@dataclass
class CombinedLossOutput:
    loss1: Tensor
    loss2: Tensor


@dataclass
class DiscriminatorLoss:
    avg: float


@dataclass
class GeneratorLoss:
    avg: float
    bce: float
    mse: float
    vgg: float


@dataclass
class MetricResult:
    discriminator: DiscriminatorLoss
    generator: GeneratorLoss


@dataclass  # src.inference.consumer
class InferenceConfig:
    model: nn.Module
    weight: str
    device: device
    batch_size: int
    input_dir: str
    target_dir: str
