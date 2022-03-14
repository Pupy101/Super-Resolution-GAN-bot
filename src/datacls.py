from dataclasses import dataclass
from typing import Any, List

from torch import nn, Tensor
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
    main: T.Compose
    resize: T.Resize


@dataclass  # train.train
class GANParameters:
    generator: Any
    discriminator: Any


@dataclass
class Model(GANParameters):
    generator: nn.Module
    discriminator: nn.Module


@dataclass
class Optimizer(GANParameters):
    generator: nn.Module
    discriminator: nn.Module


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
