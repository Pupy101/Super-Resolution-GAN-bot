"""Module with usefull dataclasses."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


@dataclass  # data.augmentation
class Augmentation:
    """Parameters for train/valid aigmentations."""

    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    main: T.Compose
    large: T.Compose
    small: T.Compose


@dataclass
class InferenceAugmentation:
    """Parameters for inference preprocessing images."""

    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    main: T.Compose
    resize: T.Resize


@dataclass  # train.train
class GANParameters:
    """Base class for model parameter."""

    generator: Any
    discriminator: Any


@dataclass
class GANModule(GANParameters):
    """Class with torch model."""

    generator: Union[nn.Module, optim.Optimizer]
    discriminator: Union[nn.Module, optim.Optimizer]

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

    def to(self, device: torch.device) -> "GANModule":
        """
        Move to another device.

        Parameters
        ----------
        device : target device

        Returns
        -------
        instance on target device
        """
        self.generator.to(device)
        self.discriminator.to(device)
        return self


@dataclass
class GANModel(GANModule):
    """Class with torch model."""

    generator: nn.Module
    discriminator: nn.Module


@dataclass
class Optimizer(GANModule):
    """Optimizers for generator and discriminator."""

    generator: optim.Optimizer
    discriminator: optim.Optimizer


@dataclass
class CriterionGenerator:
    """Criterions for generator with VGGLos + MSELoss and BCELoss."""

    vgg: nn.Module
    mse: nn.Module
    bce: nn.Module


@dataclass
class LossCoefficients:
    vgg: float
    mse: float
    bce: float


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

    input_image_size: int
    model: nn.Module
    weight: Union[str, Path]
    device: Union[torch.device, str]
    batch_size: int
    input_dir: Union[str, Path]
    target_dir: Union[str, Path]
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None


class ForwardType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
