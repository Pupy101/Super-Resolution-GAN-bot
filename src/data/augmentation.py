"""Module with augmentations."""

from typing import Optional, Tuple

from torchvision import transforms as T

from ..datacls import Augmentation, ForwardType, InferenceAugmentation


def create_augmentation(
    mode: str,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    large_image_size: int = 224,
    ratio_small_to_large: float = 0.5,
) -> Augmentation:
    """
    Function for create augmentation for train/validation.

    Args:
        mode (str): mode of forward network "train" or "validation"
        mean (Optional[Tuple[float, float, float]], optional): mean to standartization input image
        std (Optional[Tuple[float, float, float]], optional): std to standartization input image
        large_image_size (int, optional): size of output from network image size
        ratio_small_to_large (float, optional): ratio between large and small image

    Raises:
        RuntimeError: if node incorrect

    Returns:
        Augmentation: augmentation
    """
    assert ratio_small_to_large < 1, "ratio must be smaller than 1"
    if mean is None:
        mean = (0.5, 0.5, 0.5)
    if std is None:
        std = (0.5, 0.5, 0.5)
    if mode == ForwardType.TRAIN.value:
        transform = T.Compose(
            [
                T.RandomCrop((large_image_size, large_image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply(
                    [T.ColorJitter(brightness=0.1, hue=0.1), T.RandomEqualize()]
                ),
            ]
        )
    elif mode == ForwardType.VALIDATION.value:
        transform = T.Compose([T.CenterCrop((large_image_size, large_image_size))])
    else:
        raise RuntimeError(f'Strange augmentation mode: "{mode}"')
    large_image_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    small_image_size = round(large_image_size * ratio_small_to_large)
    small_image_transform = T.Compose(
        [T.Resize(small_image_size), T.ToTensor(), T.Normalize(mean=mean, std=std)]
    )
    return Augmentation(
        mean=mean,
        std=mean,
        main=transform,
        large=large_image_transform,
        small=small_image_transform,
    )


def create_inference_augmentation(
    input_image_size: int = 512,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> InferenceAugmentation:
    """
    Function for create augmentation for inference.

    Args:
        input_image_size (int, optional): input to network image size
        mean (Optional[Tuple[float, float, float]], optional): mean to standartization input image
        std (Optional[Tuple[float, float, float]], optional): std to standartization input image

    Returns:
        InferenceAugmentation: augmentation
    """
    if mean is None:
        mean = (0.5, 0.5, 0.5)
    if std is None:
        std = (0.5, 0.5, 0.5)
    return InferenceAugmentation(
        mean=mean,
        std=mean,
        main=T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)]),
        resize=T.Resize(input_image_size),
    )
