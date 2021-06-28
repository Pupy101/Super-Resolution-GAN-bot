import os

import torch

from PIL import Image
from torchvision import transforms as tt

mean, std = 0.5, 0.5


transform = tt.Compose(
    [
     tt.ToTensor(),
     tt.Normalize(mean, std)
    ]
)


def denormolize(img, mean=0.5, std=0.5):
    return img * std + mean


def preprocessing_image(dir, net, device):
    if not os.path.exists('./result/'):
        os.mkdir('result')
    images = os.listdir(dir)
    image = Image.open(images[0])
    os.remove(os.path.join(dir, images[0]))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = denormolize(net(image).squeeze(0).permute(1, 2, 0))
    return output
