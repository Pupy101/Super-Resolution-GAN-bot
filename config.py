import torch

from src.datacls import InferenceConfig
from src.model import SuperResolutionGenerator

token = "api_token"

config = InferenceConfig(
    input_image_size=512,
    model=SuperResolutionGenerator(n_increase=2),
    weight="path_to_weights",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=8,
    input_dir="/content/downloaded_images",
    target_dir="/content/uploading_images",
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)
