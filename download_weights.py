import os

import gdown

weights = {
    'url': 'https://drive.google.com/uc?id=1LQL8vtSBbemaNwwkJFJAINzNVKheIv93',
    'output': 'generator.pt',
    'quiet': False
    }
if not os.path.exists('.//pretrained_models'):
    os.mkdir('pretrained_models')
os.chdir('.//pretrained_models')
gdown.download(**weights)
