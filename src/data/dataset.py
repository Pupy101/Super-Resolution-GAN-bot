"""Module with torch dataset for super resolution network."""

from logging import getLogger
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..datacls import Augmentation

logger = getLogger(__file__)


class SuperResolutionDataset(Dataset):
    """Super resolution dataset."""

    def __init__(
        self,
        dirs: List[Union[str, Path]],
        transform: Augmentation,
        available_extensions: Optional[Set[str]] = None,
    ) -> None:
        """
        Init method.

        Parameters
        ----------
        dirs : directories with images.
        transform : Augmentation for perprocessing image and transform
        it into small and large example.
        """
        if available_extensions is None:
            available_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
        self.files = self._find_all_images(
            dirs, available_extensions=available_extensions
        )
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
    def _find_all_images(
        folders: List[Union[str, Path]], available_extensions: Set[str]
    ) -> List[Path]:
        """
        Find all images in given directories.

        Parameters
        ----------
        folders : List of directories.

        Returns
        -------
        List of path to images
        """
        images = []
        for folder in folders:
            folder = Path(folder)
            count = 0
            for file in folder.glob("*"):
                if file.suffix.lower() in available_extensions:
                    images.append(file)
                    count += 1
            logger.info('Found %s images in directory "%s"', count, folder)
        logger.info("Overall found image count is %s", len(images))
        return images

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns
        -------
        Number of training examples.
        """
        return len(self.files)
