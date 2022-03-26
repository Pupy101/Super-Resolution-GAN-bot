from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union


from numpy import ndarray
from PIL import Image
from torch import nn, tensor, Tensor

from src.data.augmentation import inference


def denormolize(
    image: Tensor,
    mean: Optional[Union[float, Tuple[float, float, float]]] = 0.5,
    std: Optional[Union[float, Tuple[float, float, float]]] = 0.5,
) -> List[ndarray]:
    image = image.detach().cpu()
    if isinstance(std, (float, int)):
        image *= std
    else:
        image *= tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    if isinstance(mean, (float, int)):
        image += mean
    else:
        image += tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    denormalized = []
    for i in range(image.size(0)):
        denormalized.append(image[i].permute(1, 2, 0).numpy())
    return denormalized


def prepare_image(path_to_image: str) -> Tensor:
    image = Image.open(path_to_image)
    image = inference.main(image)
    if image.size(1) > 512 or image.size(2) > 512:
        image = inference.resize(image)
    return image.unsqueeze(0)


def write_image(image: ndarray, target_path: Union[str, Path]) -> None:
    image = Image.fromarray(image)
    image.save(target_path)


@torch.no_grad()
def upsample_image_torch(model: nn.Module, image: Tensor) -> ndarray:
    image = image.unsqueeze(0).to(model.device)
    super_image = model(image)
    image = denormolize(super_image.cpu())
    return image


@torch.no_grad()
def upsample_image_numpy(model: nn.Module, image: Tensor) -> ndarray:
    image = image.unsqueeze(0).to(model.device)
    super_image = model(image)
    image = denormolize(super_image.cpu())


def upsample_image_file(
    model: nn.Module, image: str, valid_transform: Callable
) -> ndarray:
    image = Image.open(image)
    image = valid_transform(image)
    image = upsample_image_torch(model, image)
    return image
