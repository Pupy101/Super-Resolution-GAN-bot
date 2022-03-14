from pathlib import Path
from typing import List, Optional, Tuple, Union


from numpy import ndarray
from PIL import Image
from torch import tensor, Tensor

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
