"""Module with usefull dataclasses."""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


@dataclass  # data.augmentation
class Augmentation:
    """Parameters for train/valid aigmentations."""

    mean: List[float]
    std: List[float]
    main: T.Compose
    large: T.Compose
    small: T.Compose


@dataclass
class InferenceAugmentation:
    """Parameters for inference preprocessing images."""

    mean: List[float]
    std: List[float]
    main: T.Compose
    resize: T.Resize


@dataclass  # train.train
class GANParameters:
    """Base class for model parameter."""

    generator: Any
    discriminator: Any


@dataclass
class TorchModuleSubclass(GANParameters):
    """Class for save and load combinated model (generator/discriminator)."""

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Save model.

        Returns
        -------
        dict with keys 'generator' and 'discriminator'
        """
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def load_state_dict(self, weight: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Load model.

        Parameters
        ----------
        weight : dict with keys 'generator' and 'discriminator'
        """
        assert "generator" in weight, "Not founded 'generator' in weights"
        assert "discriminator" in weight, "Not founded 'discriminator' in weights"
        self.generator.load_state_dict(weight["generator"])
        self.discriminator.load_state_dict(weight["discriminator"])


@dataclass
class Model(TorchModuleSubclass):
    """Combinated model with generator and discriminator."""

    generator: nn.Module
    discriminator: nn.Module


@dataclass
class Optimizer(TorchModuleSubclass):
    """Optimizers for generator and discriminator."""

    generator: optim.Optimizer
    discriminator: optim.Optimizer


@dataclass
class CriterionGenerator:
    """Criterions for generator with VGGLos + MSELoss and BCELoss."""

    mse_vgg: nn.Module
    bce: nn.Module


@dataclass
class Criterion(GANParameters):
    """Critetions for generator and discriminator."""

    generator: CriterionGenerator
    discriminator: nn.Module


@dataclass
class Datasets:
    """Train and validation datasets."""

    train: Dataset
    valid: Dataset


@dataclass
class Dataloaders:
    """Train and validation dataloaders."""

    train: DataLoader
    valid: DataLoader


@dataclass
class CombinedLossOutput:
    """Losses values from combinated loss."""

    loss1: torch.Tensor
    loss2: torch.Tensor


@dataclass
class DiscriminatorLoss:
    """Loss metric from training discriminator."""

    avg: float


@dataclass
class GeneratorLoss:
    """Loss metric from training generator."""

    avg: float
    bce: float
    mse: float
    vgg: float


@dataclass
class MetricResult:
    """Loss metric from training generator and discriminator."""

    discriminator: DiscriminatorLoss
    generator: GeneratorLoss


@dataclass  # src.inference.consumer
class InferenceConfig:
    """Config for starting inference consumer."""

    model: nn.Module
    weight: str
    device: torch.device
    batch_size: int
    input_dir: str
    target_dir: str
