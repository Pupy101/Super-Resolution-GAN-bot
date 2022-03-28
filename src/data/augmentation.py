"""Module with augmentations."""

from torchvision import transforms as T

from src.datacls import Augmentation, InferenceAugmentation


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

train_image_transform = T.Compose(
    [
        T.Resize(512),
        T.RandomCrop((448, 448)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply(
            [
                T.ColorJitter(brightness=0.1, hue=0.1),
                T.RandomEqualize(),
            ]
        ),
    ]
)
valid_image_transform = T.Compose(
    [
        T.Resize(512),
        T.RandomCrop((448, 448)),
    ]
)
large_image_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ]
)
small_image_transform = T.Compose(
    [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ]
)


train = Augmentation(
    mean=MEAN,
    std=STD,
    main=train_image_transform,
    large=large_image_transform,
    small=small_image_transform,
)

valid = Augmentation(
    mean=MEAN,
    std=STD,
    main=valid_image_transform,
    large=large_image_transform,
    small=small_image_transform,
)

inference = InferenceAugmentation(
    mean=MEAN,
    std=STD,
    main=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]),
    resize=T.Resize(512),
)
