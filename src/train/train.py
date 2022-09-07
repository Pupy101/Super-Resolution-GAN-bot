"""Modile with train loop."""


from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm.notebook import tqdm

from ..datacls import (
    Criterion,
    Dataloaders,
    DiscriminatorLoss,
    GANModel,
    GeneratorLoss,
    LossCoefficients,
    MetricResult,
    Optimizer,
    Scheduler,
)
from ..utils.misc import get_patch


def train_model(
    n_epoch: int,
    model: GANModel,
    loaders: Dataloaders,
    critetion: Criterion,
    optimizer: Optimizer,
    device: torch.device,
    coefficients: LossCoefficients,
    scheduler: Optional[Scheduler] = None,
    accumulation: int = 1,
    patch_gan: bool = False,
    input_size: int = 112,
):
    """
    Train super resolution model.

    Args:
        n_epoch (int): count epoch
        model (GANModel): model with generator and discriminator
        loaders (Dataloaders): train and validation loaders
        critetion (Criterion): criterions with loss function for generator and discriminator
        optimizer (Optimizer): optimizers for generator and discriminator
        device (torch.device): device for training
        coefficients (LossCoefficients): coefficient for losses
        scheduler (Optional[Scheduler], optional): optimizer lr scheduler
        accumulation (int, optional): count of accumulation steps
        patch_gan (bool, optional): use only patch from image to compute gan
        input_size (int, optional): inout to generator image size
    """
    min_eval_loss = float("inf")
    best_epoch = 0
    save_path = Path.cwd() / "weights"
    save_path.mkdir(exist_ok=True)
    model.to(device)
    print("GENERATOR:")
    summary(model=model.generator, input_size=(3, input_size, input_size))
    print("\nDISCRIMINATOR:")
    summary(model=model.discriminator, input_size=(3, input_size, input_size))
    for i in range(n_epoch):
        print(f"EPOCH {i+1}/{n_epoch}")
        train_metric = forward_one_epoch(
            model=model,
            loader=loaders.train,
            criterion=critetion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            coefficients=coefficients,
            accumulation=accumulation,
            patch_gan=patch_gan,
            is_train=True,
        )
        train_avg_loss = train_metric.generator.avg
        eval_metric = forward_one_epoch(
            model=model,
            loader=loaders.valid,
            criterion=critetion,
            device=device,
            coefficients=coefficients,
            accumulation=accumulation,
            patch_gan=patch_gan,
            is_train=False,
        )
        eval_avg_loss = eval_metric.generator.avg
        torch.save(model.state_dict(), save_path / f"Model_{i+1}.pth")
        torch.save(optimizer.state_dict(), save_path / f"Optimizer_{i+1}.pth")
        if eval_avg_loss < min_eval_loss:
            best_epoch = i + 1
            min_eval_loss = eval_avg_loss

    print(f"Train loss:              {train_avg_loss:10.5f}")
    print(f"Eval loss:               {eval_avg_loss:10.5f}")
    print(f"Best epoch:              {best_epoch}")
    print(f"Eval loss on best epoch: {min_eval_loss:10.5f}")


def forward_one_epoch(
    model: GANModel,
    loader: DataLoader,
    criterion: Criterion,
    device: torch.device,
    coefficients: LossCoefficients,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Scheduler] = None,
    accumulation: int = 1,
    patch_gan: bool = False,
    is_train: bool = True,
) -> MetricResult:
    """
    One epoch train model.

    Args:
        model (GANModel): models with generator and discriminator
        loader (DataLoader): training loader
        criterion (Criterion): criterions with losses function for generator and discriminator
        optimizer (Optimizer): optimizers for generator and discriminator
        scheduler (Optional[Scheduler], optional): optimizer lr scheduler
        device (torch.device): device for training
        coefficients (LossCoefficients): coefficient for generator losses
        accumulation (int, optional): count of accumulation steps
        patch_gan (bool, optional): use only patch from image to compute gan

    Returns:
        MetricResult: average train metrics
    """
    context_manager = nullcontext if is_train else torch.no_grad

    ovr_loss_dis = 0.0
    ovr_loss_mse = 0.0
    ovr_loss_vgg = 0.0
    ovr_loss_bce = 0.0

    if is_train:
        model.discriminator.train()
        model.generator.train()

        if optimizer is not None:
            optimizer.discriminator.zero_grad()
            optimizer.generator.zero_grad()

        loss_dis = torch.tensor(0.0, device=device)
        loss_gen = torch.tensor(0.0, device=device)
    else:
        model.discriminator.eval()
        model.generator.eval()

    length_dataloader = len(loader)

    i = 1

    large_image: Tensor
    small_image: Tensor

    for i, (large_image, small_image) in tqdm(
        enumerate(loader, 1), leave=False, total=length_dataloader
    ):
        large_image = large_image.to(device)
        small_image = small_image.to(device)

        # generator
        # mse and vgg loss
        with context_manager():
            output_gen: Tensor = model.generator(small_image)

        loss_mse: Tensor = (
            criterion.generator.mse(output_gen, large_image) * coefficients.mse
        )
        loss_vgg: Tensor = (
            criterion.generator.vgg(output_gen, large_image) * coefficients.vgg
        )

        ovr_loss_mse += loss_mse.item()
        ovr_loss_vgg += loss_vgg.item()

        # bce loss
        label = torch.ones(small_image.size(0), device=device, dtype=torch.long)

        with context_manager():
            output_gen = model.generator(small_image)
            output_dis = model.discriminator(output_gen)

        loss_bce: Tensor = criterion.generator.bce(output_dis, label) * coefficients.bce

        ovr_loss_bce += loss_bce.item()

        if is_train:
            loss_gen += loss_vgg + loss_mse + loss_bce
            if accumulation == 1 or i % accumulation == 0 or i == length_dataloader:
                loss_gen.backward()
                if optimizer is not None:
                    optimizer.generator.step()
                    optimizer.generator.zero_grad()
                loss_gen = torch.tensor(0.0, device=device)
                if scheduler is not None:
                    scheduler.generator.step()

        # discriminator
        # real images
        label = torch.ones(small_image.size(0), device=device, dtype=torch.long)

        if patch_gan:
            patched_images, index_height, index_weight = get_patch(large_image)
        else:
            patched_images = large_image

        with context_manager():
            output_dis = model.discriminator(patched_images)

        loss_real: Tensor = criterion.discriminator(output_dis, label)

        ovr_loss_dis += loss_real.item()

        # fake images
        label = torch.zeros(small_image.size(0), device=device, dtype=torch.long)

        with torch.no_grad():
            fake_image: Tensor = model.generator(small_image)

        if patch_gan:
            patched_images, *_ = get_patch(fake_image, index_height, index_weight)
        else:
            patched_images = fake_image

        with context_manager():
            output_dis = model.discriminator(patched_images)

        loss_fake: Tensor = criterion.discriminator(output_dis, label)

        ovr_loss_dis += loss_fake.item()

        if is_train:
            loss_dis += loss_fake + loss_real
            if accumulation == 1 or i % accumulation == 0 or i == length_dataloader:
                loss_dis.backward()
                if optimizer is not None:
                    optimizer.discriminator.step()
                    optimizer.discriminator.zero_grad()
                loss_dis = torch.tensor(0.0, device=device)
                if scheduler is not None:
                    scheduler.discriminator.step()

    avg_loss_dis = ovr_loss_dis / i
    avg_loss_gen = (ovr_loss_bce + ovr_loss_mse + ovr_loss_vgg) / i
    avg_loss_bce = ovr_loss_bce / i
    avg_loss_mse = ovr_loss_mse / i
    avg_loss_vgg = ovr_loss_vgg / i

    print("TRAIN:" if is_train else "EVAL:")
    print(f"\tDiscriminator Loss: {avg_loss_dis:7.3f}")
    print(f"\tGenerator Loss:     {avg_loss_gen:7.3f}")
    print(f"\tGenerator GANLoss:  {avg_loss_bce:7.3f}")
    print(f"\tGenerator MSELoss:  {avg_loss_mse:7.3f}")
    print(f"\tGenerator VGGLoss:  {avg_loss_vgg:7.3f}")
    return MetricResult(
        discriminator=DiscriminatorLoss(avg=avg_loss_dis),
        generator=GeneratorLoss(
            avg=avg_loss_gen, bce=avg_loss_bce, mse=avg_loss_mse, vgg=avg_loss_vgg
        ),
    )
