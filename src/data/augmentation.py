"""Module with augmentations."""

from typing import Optional, Tuple

from torchvision import transforms as T

from ..datacls import Augmentation, InferenceAugmentation


def create_train_augmentation(
    train: bool = True,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    large_image_size: int = 448,
    ratio_small_to_large: float = 0.5,
) -> Augmentation:
    assert ratio_small_to_large < 1, "ratio must be smaller than 1"
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    if train:
        transform = T.Compose(
            [
                T.Resize(large_image_size + 64),
                T.RandomCrop((large_image_size, large_image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply(
                    [T.ColorJitter(brightness=0.1, hue=0.1), T.RandomEqualize()]
                ),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize(large_image_size + 64),
                T.RandomCrop((large_image_size, large_image_size)),
            ]
        )
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
) -> Augmentation:
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    return InferenceAugmentation(
        mean=mean,
        std=mean,
        main=T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)]),
        resize=T.Resize(input_image_size),
    )
