"""Module with functions."""

from pathlib import Path
from random import randint
from typing import Generator, Iterable, List, Optional, Tuple, TypeVar, Union

import torch
from numpy import ndarray
from PIL import Image
from src.model import SuperResolutionGenerator
from torch import Tensor

from ..datacls import InferenceAugmentation

T = TypeVar("T")


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
    denormalized images as List[numpy.ndarray]
    """
    image = image.detach().cpu()
    if isinstance(std, float):
        image *= std
    else:
        image *= torch.tensor(std)[None, :, None, None]
    if isinstance(mean, float):
        image += mean
    else:
        image += torch.tensor(mean)[None, :, None, None]
    denormalized = []
    for i in range(image.size(0)):
        denormalized.append(image[i].permute(1, 2, 0).numpy())
    return denormalized


def prepare_image(
    path_to_image: Union[str, Path], augmentation: InferenceAugmentation
) -> Tensor:
    """
    Preprocessing image for network inference.

    Parameters
    ----------
    path_to_image : path to image
    augmentation: augmentation

    Returns
    -------
    image as torch.Tensor
    """
    image = Image.open(path_to_image)
    preprocessed = augmentation.main(image)
    if image.size(1) > 512 or image.size(2) > 512:
        preprocessed_resized: Tensor = augmentation.resize(preprocessed)
    return preprocessed_resized.unsqueeze(0)


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
def upsample_images_torch(
    model: SuperResolutionGenerator, images: Tensor
) -> List[ndarray]:
    """
    Upsample image from torch.Tensor.

    Parameters
    ----------
    model : super resolution generator
    image : input image or images as torch.Tensor

    Returns
    -------
    preprocessed images as List[numpy.ndarray]
    """
    if len(images.shape) < 4:
        images = images.unsqueeze(0)
    images = images.to(model.device)
    upsampled_image = model.forward(images)
    preprocessed = denormolize(upsampled_image.cpu())
    return preprocessed


@torch.no_grad()
def upsample_images_numpy(
    model: SuperResolutionGenerator, images: ndarray
) -> List[ndarray]:
    """
    Upsample image from numpy.ndarray.

    Parameters
    ----------
    model : super resolution generator
    image : input image as numpy.ndarray

    Returns
    -------
    preprocessed images as List[numpy.ndarray]
    """
    torch_images = torch.from_numpy(images)
    preprocessed = upsample_images_torch(model=model, images=torch_images)
    return preprocessed


def upsample_images_file(
    model: SuperResolutionGenerator,
    augmentation: InferenceAugmentation,
    path_to_images: Union[Union[str, Path], List[Union[str, Path]]],
) -> List[ndarray]:
    """
    Upsample image from file.

    Parameters
    ----------
    model : super resolution generator
    augmentation : augmentation
    path_to_image : path to image or images

    Returns
    -------
    preprocessed images as List[numpy.ndarray]
    """
    if isinstance(path_to_images, (Path, str)):
        image = prepare_image(path_to_images, augmentation=augmentation)
    else:
        images = []
        for path_to_image in path_to_images:
            image = prepare_image(path_to_image, augmentation=augmentation)
            images.append(image)
        image = torch.cat(images, dim=0)
    return upsample_images_torch(model, image)


def get_patch(
    images: Tensor,
    height_index: Optional[int] = None,
    weight_index: Optional[int] = None,
    patch_size: int = 224,
) -> Tuple[Tensor, int, int]:
    """
    Get patch from images.

    Parameters
    ----------
    images : batch of images
    height_index : index height for patch
    weight_index : index weight for patch

    Returns
    -------
    random pathes from images
    """
    height, weight = images.size(2), images.size(3)
    max_height, max_weight = height - patch_size - 1, weight - patch_size - 1
    h_i = height_index
    w_i = weight_index
    if h_i is None:
        h_i = randint(0, max_height)
    if w_i is None:
        w_i = randint(0, max_weight)

    patch = images[:, :, h_i : h_i + patch_size, w_i : w_i + patch_size]
    return patch, h_i, w_i


def create_chunks(
    items: Iterable[T], chunk_size: int
) -> Generator[List[T], None, None]:
    chunk: List[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
