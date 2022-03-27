"""Consumer class for preprocessing image in another process."""

from math import ceil
from pathlib import Path
from time import sleep
from typing import List

import torch

from src.datacls import InferenceConfig
from src.utils.misc import denormolize, prepare_image, write_image


class SuperResolutionConsumer:
    """Class for inference image from super resolution network."""

    def __init__(
        self,
        config: InferenceConfig,
    ):
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
        self.b_s = config.weight
        self.input_dir = Path(config.input_dir)
        self.target_dir = Path(config.target_dir)
        assert (
            self.input_dir.is_dir() and self.target_dir.is_dir()
        ), "Check input and target path model consumer"

    def prepare_model(self, weight: str) -> None:
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
        images = []
        for file in self.input_dir.glob("*.*"):
            if file.suffix not in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
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
        prepared = []
        for path in batch:
            prepared.append(prepare_image(path))
        prepared_batch = torch.cat(prepared, dim=0)
        with torch.no_grad():
            output = self.model(prepared_batch.to(self.device))
        denormalized = denormolize(output)
        for i, image in enumerate(denormalized):
            target = self.target_dir / batch[i].name
            write_image(image, str(target))

    def run(self) -> None:
        """
        Run consumer.

        It every second check input dir.
        If there is an image in the directory,
        it forms a batch and conducts them through the network.
        """
        while True:
            input_images = self.check_input_dir()
            for i in range(ceil(len(input_images) / self.b_s)):
                batch = input_images[i * self.b_s : (i + 1) * self.b_s]
                self.handle_batch(batch)
            sleep(1)
