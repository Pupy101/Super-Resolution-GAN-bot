from torchvision import transforms as T

from src.datacls import Augmentation, InferenceAugmentation


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

train_image_transform = T.Compose(
    [
        T.Resize(270),
        T.RandomCrop((224, 224)),
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
        T.Resize(270),
        T.CenterCrop((224, 224)),
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
        T.Resize(112),
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
    main=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]),
    resize=T.Resize(512),
)
