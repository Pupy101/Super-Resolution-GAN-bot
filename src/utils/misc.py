"""Module with functions."""

from pathlib import Path
from random import randint
from typing import List, Optional, Tuple, Union

import torch

from numpy import ndarray
from PIL import Image
from torch import Tensor

from src.data.augmentation import inference
from src.model import SuperResolutionGenerator


def denormolize(
    image: Tensor,
    mean: Optional[Union[float, Tuple[float, float, float]]] = 0.5,
    std: Optional[Union[float, Tuple[float, float, float]]] = 0.5,
) -> List[ndarray]:
    """
    Denormalize image from network.

    Parameters
    ----------
    image : output images from net
    mean : mean
    std : std

    Returns
    -------
    list of denormalized images as numpy.ndarray work
    """
    image = image.detach().cpu()
    if isinstance(std, (float, int)):
        image *= std
    else:
        image *= torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    if isinstance(mean, (float, int)):
        image += mean
    else:
        image += torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    denormalized = []
    for i in range(image.size(0)):
        denormalized.append(image[i].permute(1, 2, 0).numpy())
    return denormalized


def prepare_image(path_to_image: Union[str, Path]) -> Tensor:
    """
    Preprocessing image for network inference.

    Parameters
    ----------
    path_to_image : path to image

    Returns
    -------
    image as torch.Tensor
    """
    image = Image.open(path_to_image)
    image = inference.main(image)
    if image.size(1) > 512 or image.size(2) > 512:
        image = inference.resize(image)
    return image.unsqueeze(0)


def write_image(image: ndarray, target_path: Union[str, Path]) -> None:
    """
    Write image to file.

    Parameters
    ----------
    image : image as numpy.ndarray
    target_path : target to output file
    """
    pill_image = Image.fromarray(image)
    pill_image.save(target_path)


@torch.no_grad()
def upsample_image_torch(model: SuperResolutionGenerator, image: Tensor) -> ndarray:
    """
    Upsample image from torch.Tensor.

    Parameters
    ----------
    model : super resolution generator
    image : input image as torch.Tensor

    Returns
    -------
    preprocessed image
    """
    if len(image.shape) < 4:
        image = image.unsqueeze(0)
    image = image.to(model.device)
    super_image = model(image)
    preprocessed = denormolize(super_image.cpu())
    return preprocessed[0]


@torch.no_grad()
def upsample_image_numpy(model: SuperResolutionGenerator, image: Tensor) -> ndarray:
    """
    Upsample image from numpy.ndarray.

    Parameters
    ----------
    model : super resolution generator
    image : input image as numpy.ndarray

    Returns
    -------
    preprocessed image
    """
    image = image.unsqueeze(0).to(model.device)
    super_image = model(image)
    preprocessed = denormolize(super_image.cpu())
    return preprocessed[0]


def upsample_image_file(model: SuperResolutionGenerator, path_to_image: str) -> ndarray:
    """
    Upsample image from file.

    Parameters
    ----------
    model : super resolution generator
    path_to_image : path to image
    valid_transform : Callable
        _description_

    Returns
    -------
    ndarray
        _description_
    """
    image = prepare_image(path_to_image)
    return upsample_image_torch(model, image)


def get_patch(
    images: Tensor,
    indxes_height: Optional[List[int]] = None,
    indxes_weight: Optional[List[int]] = None,
) -> Tuple[Tensor, List[int], List[int]]:
    """
    Get patch from images.

    Parameters
    ----------
    images : batch of images
    indxes_height : indexes height for patch
    indxes_weight : indexes weight for patch

    Returns
    -------
    random pathes from images
    """
    batch_size = images.size(0)
    height, weight = images.size(2), images.size(3)
    max_height, max_weight = height - 225, weight - 225
    batch = []
    if indxes_height is None:
        indxes_height = [randint(0, max_height) for _ in range(batch_size)]
    if indxes_weight is None:
        indxes_weight = [randint(0, max_weight) for _ in range(batch_size)]

    for i in range(batch_size):
        batch.append(
            images[
                i : i + 1,
                :,
                indxes_height[i] : indxes_height[i] + 224,
                indxes_weight[i] : indxes_weight[i] + 224,
            ]
        )
    return torch.cat(batch), indxes_height, indxes_weight
