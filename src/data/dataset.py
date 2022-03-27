"""Module with torch dataset for super resolution network."""

import os

from os.path import join as join_path
from typing import List, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.datacls import Augmentation


class SuperResolutionDataset(Dataset):
    """Super resolution dataset."""

    def __init__(self, dirs: List[str], transform: Augmentation):
        """
        Init method.

        Parameters
        ----------
        dirs : directories with images.
        transform : Augmentation for perprocessing image and transform
        it into small and large example.
        """
        self.files = self._find_all_images(dirs)
        self.transforms = transform

    def __getitem__(self, ind: int) -> Tuple[Tensor, Tensor]:
        """
        Get item method.

        Parameters
        ----------
        ind : index of pair large and small image.

        Returns
        -------
        Pair large and small image.
        """
        image_path = self.files[ind]
        image = self.transforms.main(Image.open(image_path))
        image_copy = image.copy()
        large_image = self.transforms.large(image)
        small_image = self.transforms.small(image_copy)
        return large_image, small_image

    @staticmethod
    def _find_all_images(paths_to_directories: List[str]) -> List[str]:
        """
        Find all images in given directories.

        Parameters
        ----------
        paths_to_directories : List of directories.

        Returns
        -------
        List of path to images
        """
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
        """
        Get dataset length.

        Returns
        -------
        Number of training examples.
        """
        return len(self.files)
