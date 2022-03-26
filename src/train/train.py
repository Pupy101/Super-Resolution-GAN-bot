import os

from os.path import join as join_path

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datacls import (
    Criterion,
    Dataloaders,
    DiscriminatorLoss,
    GeneratorLoss,
    MetricResult,
    Model,
    Optimizer,
)


def train_model(
    n_epoch: int,
    model: Model,
    loaders: Dataloaders,
    critetion: Criterion,
    optimizer: Optimizer,
    device: torch.device,
):
    min_eval_loss = float("inf")
    os.makedirs("weights", exist_ok=True)
    for i in range(n_epoch):
        print(f"\tEpoch {i+1}/{n_epoch}")
        train_metric: MetricResult = train_one_epoch(
            model=model,
            loader=loaders.train,
            criterion=critetion,
            optimizer=optimizer,
            device=device,
        )
        train_avg_loss = train_metric.generator.avg
        eval_metric: MetricResult = evaluate_one_epoch(
            model=model, loader=loaders.valid, criterion=critetion, device=device
        )
        eval_avg_loss = eval_metric.generator.avg
        if train_avg_loss > eval_avg_loss and eval_avg_loss < min_eval_loss:
            min_eval_loss = eval_avg_loss
            torch.save(
                model.discriminator.state_dict(),
                join_path("./weights", f"Model_DIS_{i+1}.pth"),
            )
            torch.save(
                model.generator.state_dict(),
                join_path("./weights", f"Model_GEN_{i+1}.pth"),
            )
    print(
        f"Best metric Train loss: {train_avg_loss:10.5f}\t"
        f"Eval loss: {eval_avg_loss:10.5f}"
    )


def train_one_epoch(
    model: Model,
    loader: DataLoader,
    criterion: Criterion,
    optimizer: Optimizer,
    device: torch.device,
) -> MetricResult:
    avg_loss_gen = 0
    avg_loss_dis = 0
    avg_loss_mse = 0
    avg_loss_vgg = 0
    avg_loss_bce = 0
    count = 0

    model.discriminator.train()
    model.generator.train()

    for large_image, small_image in tqdm(loader, leave=False):
        count += 1
        large_image = large_image.to(device)
        small_image = small_image.to(device)

        # discriminator
        optimizer.discriminator.zero_grad()
        # real images
        label = torch.ones(small_image.size(0), device=device, dtype=torch.long)
        output_dis = model.discriminator(large_image)
        real_loss = criterion.discriminator(output_dis, label)
        avg_loss_dis += real_loss.item()
        # fake images
        label = torch.zeros(small_image.size(0), device=device, dtype=torch.long)
        with torch.no_grad():
            fake_image = model.generator(small_image)
        output_dis = model.discriminator(fake_image)
        fake_loss = criterion.discriminator(output_dis, label)
        avg_loss_dis += fake_loss.item()

        loss_dis = fake_loss + real_loss
        loss_dis.backward()
        optimizer.discriminator.step()

        # generator
        optimizer.generator.zero_grad()
        # mse and vgg loss
        output_gen = model.generator(small_image)
        loss_mse_vgg = criterion.generator.mse_vgg(output_gen, large_image)
        vgg_loss = loss_mse_vgg.loss1
        mse_loss = loss_mse_vgg.loss2
        avg_loss_gen += vgg_loss.item() + mse_loss.item()
        avg_loss_mse += mse_loss.item()
        avg_loss_vgg += vgg_loss.item()
        # bce loss
        label = torch.ones(small_image.size(0), device=device, dtype=torch.long)
        output_gen = model.generator(small_image)
        output_dis = model.discriminator(output_gen)
        loss_bce = criterion.generator.bce(output_dis, label)
        avg_loss_gen += 1e-3 * loss_bce.item()
        avg_loss_bce += loss_bce.item()

        loss_gen = vgg_loss + mse_loss + 1e-3 * loss_bce
        loss_gen.backward()
        optimizer.generator.step()

    print(
        f"TRAIN Discriminator Loss: {avg_loss_dis / count:7.3f}\t"
        f"Generator Loss: {avg_loss_gen / count:7.3f}\t"
        f"Gen GANLoss: {avg_loss_bce / count:7.3f}  "
        f"Gen MSELoss: {avg_loss_mse / count:7.3f}  "
        f"Gen VGGLoss: {avg_loss_vgg / count:7.3f}"
    )
    return MetricResult(
        discriminator=DiscriminatorLoss(avg=avg_loss_dis),
        generator=GeneratorLoss(
            avg=avg_loss_gen, bce=avg_loss_bce, mse=avg_loss_mse, vgg=avg_loss_vgg
        ),
    )


@torch.no_grad()
def evaluate_one_epoch(
    model: Model,
    loader: DataLoader,
    criterion: Criterion,
    device: torch.device,
) -> MetricResult:
    avg_loss_gen = 0
    avg_loss_dis = 0
    avg_loss_mse = 0
    avg_loss_vgg = 0
    avg_loss_bce = 0
    count = 0

    model.discriminator.eval()
    model.generator.eval()

    for large_image, small_image in tqdm(loader, leave=False):
        count += 1
        large_image = large_image.to(device)
        small_image = small_image.to(device)

        # discriminator
        # real images
        label = torch.ones(small_image.size(0), device=device, dtype=torch.long)
        output_dis = model.discriminator(large_image)
        real_loss = criterion.discriminator(output_dis, label)
        avg_loss_dis += real_loss.item()
        # fake images
        label = torch.zeros(small_image.size(0), device=device, dtype=torch.long)
        fake_image = model.generator(small_image)
        output_dis = model.discriminator(fake_image)
        fake_loss = criterion.discriminator(output_dis, label)
        avg_loss_dis += fake_loss.item()

        # generator
        #  mse and vgg loss
        output_gen = model.generator(small_image)
        loss_mse_vgg = criterion.generator.mse_vgg(output_gen, large_image)
        vgg_loss = loss_mse_vgg.loss1
        mse_loss = loss_mse_vgg.loss2
        avg_loss_gen += vgg_loss.item() + mse_loss.item()
        avg_loss_mse += mse_loss.item()
        avg_loss_vgg += vgg_loss.item()
        # bce loss
        label = torch.ones(small_image.size(0), device=device, dtype=torch.long)
        output_gen = model.generator(small_image)
        output_dis = model.discriminator(output_gen)
        loss_bce = criterion.generator.bce(output_dis, label)
        avg_loss_gen += 1e-3 * loss_bce.item()
        avg_loss_bce += loss_bce.item()
    print(
        f"EVAL Discriminator Loss: {avg_loss_dis / count:7.3f}\t"
        f"Generator Loss: {avg_loss_gen / count:7.3f}\t"
        f"Gen GANLoss: {avg_loss_bce / count:7.3f}  "
        f"Gen MSELoss: {avg_loss_mse / count:7.3f}  "
        f"Gen VGGLoss: {avg_loss_vgg / count:7.3f} "
    )
    return MetricResult(
        discriminator=DiscriminatorLoss(avg=avg_loss_dis),
        generator=GeneratorLoss(
            avg=avg_loss_gen, bce=avg_loss_bce, mse=avg_loss_mse, vgg=avg_loss_vgg
        ),
    )
