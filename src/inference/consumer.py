"""Consumer class for preprocessing image in another process."""

from pathlib import Path
from time import sleep
from typing import List, Union

import torch
from torch import Tensor

from ..data.augmentation import create_inference_augmentation
from ..datacls import InferenceConfig
from ..utils.misc import create_chunks, denormolize, prepare_image, write_image


class SuperResolutionConsumer:
    """Class for inference image from super resolution network."""

    def __init__(self, config: InferenceConfig) -> None:
        """
        Init method.

        Parameters
        ----------
        config : super resolution network inference config
        """
        self.config = config
        self.model = config.model
        self.device = config.device
        self.prepare_model(config.weight)
        self.augmentation = create_inference_augmentation(
            input_image_size=config.input_image_size, mean=config.mean, std=config.std
        )
        self.batch_size = config.batch_size
        self.input_dir = Path(config.input_dir)
        self.target_dir = Path(config.target_dir)
        self.available_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
        assert (
            self.input_dir.is_dir() and self.target_dir.is_dir()
        ), "Check input and target path model consumer"

    def prepare_model(self, weight: Union[str, Path]) -> None:
        """
        Load weights and transfer network on right device.

        Parameters
        ----------
        weight : path to pretrained weight
        """
        self.model.eval().to(self.device)
        self.model.load_state_dict(torch.load(weight))

    def check_input_dir(self) -> List[Path]:
        """
        Check images in input dir.

        Returns
        -------
        List of input images.
        """
        images: List[Path] = []
        for file in self.input_dir.glob("*"):
            if file.suffix.lower() not in self.available_extensions:
                file.unlink()
                continue
            images.append(file)
        return images

    def handle_batch(self, batch: List[Path]) -> None:
        """
        Handle a batch of images.

        Parameters
        ----------
        batch : List of path to images
        """
        prepared: List[Tensor] = []
        for path in batch:
            prepared.append(prepare_image(path, augmentation=self.augmentation))
        prepared_batch = torch.cat(prepared, dim=0)
        with torch.no_grad():
            output: Tensor = self.model(prepared_batch.to(self.device))
        denormalized = denormolize(output, mean=self.config.mean, std=self.config.std)
        for i, image in enumerate(denormalized):
            target = self.target_dir / batch[i].name
            write_image(image, target)

    def run(self) -> None:
        """
        Run consumer.

        It every second check input dir.
        If there is an image in the directory,
        it forms a batch and conducts them through the network.
        """
        while True:
            input_images = self.check_input_dir()
            for batch in create_chunks(input_images, chunk_size=self.batch_size):
                self.handle_batch(batch)
            sleep(1)
