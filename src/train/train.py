"""Train model."""


from pathlib import Path

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
    accumulation: int = 1,
    patch_gan: bool = False,
    input_size: int = 224,
):
    """
    Train super resolution model.

    Parameters
    ----------
    n_epoch : count epoch
    model : model with generator and discriminator
    loaders : loaders with train and validation
    critetion : criterions with loss function for generator and discriminator
    optimizer : optimizers for generator and discriminator
    device : device for training
    patch_gan : use only patch from image to compute gan
    """
    min_eval_loss = float("inf")
    save_path = Path.cwd() / "weights"
    save_path.mkdir(exist_ok=True)
    print("Generator:")
    summary(model=model.generator, input_size=input_size)
    print("Discriminator:")
    summary(model=model.generator, input_size=input_size)
    for i in range(n_epoch):
        print(f"\tEpoch {i+1}/{n_epoch}")
        train_metric = train_one_epoch(
            model=model,
            loader=loaders.train,
            criterion=critetion,
            optimizer=optimizer,
            device=device,
            coefficients=coefficients,
            accumulation=accumulation,
            patch_gan=patch_gan,
        )
        train_avg_loss = train_metric.generator.avg
        eval_metric = evaluate_one_epoch(
            model=model,
            loader=loaders.valid,
            criterion=critetion,
            device=device,
            coefficients=coefficients,
            patch_gan=patch_gan,
        )
        eval_avg_loss = eval_metric.generator.avg
        if train_avg_loss > eval_avg_loss and eval_avg_loss < min_eval_loss:
            min_eval_loss = eval_avg_loss
            torch.save(model.state_dict(), save_path / f"Model_{i+1}.pth")
            torch.save(optimizer.state_dict(), save_path / f"Optimizer_{i+1}.pth")
    print(f"Best metric:")
    print("\tTrain loss: {train_avg_loss:10.5f}")
    print(f"\tEval loss: {eval_avg_loss:10.5f}")


def train_one_epoch(
    model: GANModel,
    loader: DataLoader,
    criterion: Criterion,
    optimizer: Optimizer,
    device: torch.device,
    coefficients: LossCoefficients,
    accumulation: int = 1,
    patch_gan: bool = False,
) -> MetricResult:
    """
    Train model.

    Parameters
    ----------
    model : models with generator and discriminator
    loader : training loader
    criterion : criterions with loss function for generator and discriminator
    optimizer : optimizers for generator and discriminator
    device : device for training
    patch_gan : use only patch from image to compute gan

    Returns
    -------
    all losses
    """
    ovr_loss_dis = 0
    ovr_loss_mse = 0
    ovr_loss_vgg = 0
    ovr_loss_bce = 0

    model.discriminator.train()
    model.generator.train()
    length_dataloader = len(loader)
    loss_dis = 0
    loss_gen = 0
    optimizer.discriminator.zero_grad()
    optimizer.generator.zero_grad()

    for i, (large_image, small_image) in tqdm(
        enumerate(loader, 1), leave=False, total=length_dataloader
    ):
        large_image: Tensor
        small_image: Tensor
        large_image = large_image.to(device)
        small_image = small_image.to(device)

        # generator
        # mse and vgg loss
        output_gen: Tensor = model.generator(small_image)
        loss_mse: Tensor = (
            criterion.generator.mse(output_gen, large_image) * coefficients.mse
        )
        loss_vgg: Tensor = (
            criterion.generator.vgg(output_gen, large_image) * coefficients.vgg
        )
        ovr_loss_mse += loss_vgg.item()
        ovr_loss_vgg += loss_vgg.item()
        # bce loss
        label = torch.ones(small_image.size(0), device=device, dtype=torch.long)
        output_gen = model.generator(small_image)
        output_dis = model.discriminator(output_gen)
        loss_bce: Tensor = criterion.generator.bce(output_dis, label) * coefficients.bce
        ovr_loss_bce += loss_bce.item()

        loss_gen += loss_vgg + loss_mse + loss_bce
        if accumulation == 1 or i % accumulation == 0 or i == length_dataloader:
            loss_gen.backward()
            optimizer.generator.step()
            optimizer.generator.zero_grad()
            loss_gen = 0

        # discriminator
        # real images
        label = torch.ones(small_image.size(0), device=device, dtype=torch.long)
        if patch_gan:
            patched_images, index_height, index_weight = get_patch(large_image)
        else:
            patched_images = large_image
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
        output_dis = model.discriminator(patched_images)
        loss_fake: Tensor = criterion.discriminator(output_dis, label)
        ovr_loss_dis += loss_fake.item()

        loss_dis += loss_fake + loss_real
        if accumulation == 1 or i % accumulation == 0 or i == length_dataloader:
            loss_dis.backward()
            optimizer.discriminator.step()
            optimizer.discriminator.zero_grad()
            loss_dis = 0

    avg_loss_dis = ovr_loss_dis / i
    avg_loss_gen = (ovr_loss_bce + ovr_loss_mse + ovr_loss_vgg) / i
    avg_loss_bce = ovr_loss_bce / i
    avg_loss_mse = ovr_loss_mse / i
    avg_loss_vgg = ovr_loss_vgg / i

    print("TRAIN:")
    print(f"\tDiscriminator Loss: {avg_loss_dis:7.3f}")
    print(f"\tGenerator Loss: {avg_loss_gen:7.3f}")
    print(f"\tGenerator GANLoss: {avg_loss_bce:7.3f}")
    print(f"\tGenerator MSELoss: {avg_loss_mse:7.3f}")
    print(f"\tGenerator VGGLoss: {avg_loss_vgg:7.3f}")
    return MetricResult(
        discriminator=DiscriminatorLoss(avg=avg_loss_dis),
        generator=GeneratorLoss(
            avg=avg_loss_gen, bce=avg_loss_bce, mse=avg_loss_mse, vgg=avg_loss_vgg
        ),
    )


@torch.no_grad()
def evaluate_one_epoch(
    model: GANModel,
    loader: DataLoader,
    criterion: Criterion,
    device: torch.device,
    coefficients: LossCoefficients,
    patch_gan: bool = False,
) -> MetricResult:
    """
    Validate model.

    Parameters
    ----------
    model : model with generator and discriminator
    loader : validation loader
    criterion : criterions with loss function for generator and discriminator
    device : device for validation
    patch_gan : use only patch from image to compute gan

    Returns
    -------
    all losses
    """
    ovr_loss_dis = 0
    ovr_loss_mse = 0
    ovr_loss_vgg = 0
    ovr_loss_bce = 0
    length_dataloader = len(loader)

    model.discriminator.eval()
    model.generator.eval()

    for i, (large_image, small_image) in tqdm(
        enumerate(loader, 1), leave=False, total=length_dataloader
    ):
        large_image: Tensor
        small_image: Tensor
        large_image = large_image.to(device)
        small_image = small_image.to(device)

        # generator
        #  mse and vgg loss
        output_gen = model.generator(small_image)
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
        output_gen = model.generator(small_image)
        output_dis = model.discriminator(output_gen)
        loss_bce: Tensor = criterion.generator.bce(output_dis, label) * coefficients.bce
        ovr_loss_bce += loss_bce.item()

        # discriminator
        # real images
        label = torch.ones(small_image.size(0), device=device, dtype=torch.long)
        if patch_gan:
            patched_images, index_height, index_weight = get_patch(large_image)
        else:
            patched_images = large_image
        output_dis = model.discriminator(patched_images)
        loss_real: Tensor = criterion.discriminator(output_dis, label)
        ovr_loss_dis += loss_real.item()
        # fake images
        label = torch.zeros(small_image.size(0), device=device, dtype=torch.long)
        fake_image = model.generator(small_image)
        if patch_gan:
            patched_images, *_ = get_patch(fake_image, index_height, index_weight)
        else:
            patched_images = fake_image
        output_dis = model.discriminator(patched_images)
        loss_fake: Tensor = criterion.discriminator(output_dis, label)
        ovr_loss_dis += loss_fake.item()

    avg_loss_dis = ovr_loss_dis / i
    avg_loss_gen = (ovr_loss_bce + ovr_loss_mse + ovr_loss_vgg) / i
    avg_loss_bce = ovr_loss_bce / i
    avg_loss_mse = ovr_loss_mse / i
    avg_loss_vgg = ovr_loss_vgg / i

    print("EVAL:")
    print(f"\tDiscriminator Loss: {avg_loss_dis:7.3f}")
    print(f"Generator Loss: {avg_loss_gen:7.3f}")
    print(f"Generator GANLoss: {avg_loss_bce:7.3f}")
    print(f"Gen MSELoss: {avg_loss_mse:7.3f}")
    print(f"Gen VGGLoss: {avg_loss_vgg:7.3f}")
    return MetricResult(
        discriminator=DiscriminatorLoss(avg=avg_loss_dis),
        generator=GeneratorLoss(
            avg=avg_loss_gen, bce=avg_loss_bce, mse=avg_loss_mse, vgg=avg_loss_vgg
        ),
    )
