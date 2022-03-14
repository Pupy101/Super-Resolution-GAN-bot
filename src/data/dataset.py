import os

from os.path import join as join_path
from typing import List, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.datacls import Augmentation


class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        dirs: List[str],
        transform: Augmentation = None,
    ):
        self.files = self._find_all_images(dirs)
        self.transforms = transform

    def __getitem__(self, ind: int) -> Tuple[Tensor, Tensor]:
        image_path = self.files[ind]
        image = self.transforms.main(Image.open(image_path))
        image_copy = image.copy()
        large_image = self.transforms.large(image)
        small_image = self.transforms.small(image_copy)
        return large_image, small_image

    @staticmethod
    def _find_all_images(paths_to_directories: List[str]) -> List[str]:
        images = []
        for path_to_directory in paths_to_directories:
            images.extend(
                [
                    join_path(path_to_directory, image)
                    for image in os.listdir(path_to_directory)
                ]
            )
        return images

    def __len__(self) -> int:
        return len(self.files)
