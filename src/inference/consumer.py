from math import ceil
from pathlib import Path
from time import sleep
from typing import List


from torch import cat, device, nn, load, no_grad

from src.utils.misc import denormolize, prepare_image, write_image


class SuperResolutionConsumer:
    def __init__(
        self,
        model: nn.Module,
        weight: str,
        device: device,
        batch_size: int,
        input_dir: str,
        target_dir: str,
    ):
        self.model = model
        self.device = device
        self.prepare_model(weight)
        self.batch_size = batch_size
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        assert (
            self.input_dir.is_dir() and self.target_dir.is_dir()
        ), "Check input and target path model consumer"

    def prepare_model(self, weight: str) -> None:
        self.model.eval.to(self.device)
        self.model.load_state_dict(load(weight))

    def check_input_dir(self) -> List[Path]:
        images = []
        for file in self.input_dir.glob("*.*"):
            if file.suffix not in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
                file.unlink()
                continue
            images.append(file)
        return images

    def handle_batch(self, batch: List[Path]) -> None:
        prepared = []
        for path in batch:
            prepared.append(prepare_image(path))
        prepared = cat(prepared, dim=0)
        with no_grad():
            output = self.model(prepared.to(self.device))
        denormalized = denormolize(output)
        for i in range(len(batch)):
            target = self.target_dir / batch[i].name
            write_image(denormalized[i], str(target))

    def run(self):
        while True:
            input_images = self.check_input_dir()
            for i in range(ceil(len(input_images) / self.batch_size)):
                batch = input_images[i * self.batch_size : (i + 1) * self.batch_size]
                self.handle_batch(batch)
            sleep(1)
