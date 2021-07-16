import os
import logging

import torch

from urllib.request import urlretrieve
from aiogram import Bot, Dispatcher, executor, types
from utils.data_for_bot import *
from utils.functions import preprocessing_image
from model import Generator

# Initialization of the neural network
model = Generator()
model.load_state_dict(torch.load('./pretrained_models/generator.pt'))

# Set button
button_hi = types.KeyboardButton('Привет! 👋\nРасскажи о себе')
greet_kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
greet_kb.add(button_hi)

# Get logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


# Create simple hadlers for /start text and photo
@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    global state, trigger
    state = 'start'
    trigger = True
    await message.answer("Привет! Я бот Tucker, рад что ты присоединился!", reply_markup=greet_kb)


@dp.message_handler(content_types=['text'])
async def answer(message):
    global state, trigger
    if message.text == 'Привет! 👋\nРасскажи о себе' and state == 'start':
        await message.answer(representation_bot)
        if trigger:
            state = 'process'
        else:
            state = 'end'
    elif message.text == 'Привет! 👋\nРасскажи о себе' and state == 'process':
        await message.answer(representation_bot_2)
        state = 'start'
        trigger = False
    elif message.text == 'Привет! 👋\nРасскажи о себе' and state == 'end':
        await message.answer('Я уже все рассказал и не собираюсь повторяться, просто загружай фото')
    else:
        await message.answer(message.text)


@dp.message_handler(content_types=['document', 'photo'])
async def scan_message(msg: types.Message):
    try:
        document_id = msg.document.file_id
    except:
        print(1111111111111111111111111111111111)
        document_id = msg.photo[-1].file_id
    file_info = await bot.get_file(document_id)
    fi = file_info.file_path
    name, ext = os.path.splitext(file_info['file_path'].split('/')[-1])
    urlretrieve(f'https://api.telegram.org/file/bot{API_TOKEN}/{fi}', f'./files/{str(msg.from_user.id)+ext}')
    file_name = f'{str(msg.from_user.id)+ext}'
    await bot.send_message(msg.from_user.id, 'Файл успешно сохранён и нейросеть начала его обрабатывать')
    # Preprocess image with NN
    preprocessing_image('./files', model, file_name)
    # If file in result bot send it to user and delete from folder
    if os.listdir('.//result//'):
        image = os.listdir('.//result//')[0]
        with open(os.path.join('./result', image), 'rb') as photo:
            await bot.send_document(msg.from_user.id, photo)
            os.remove(os.path.join('./result', image))


if __name__ == '__main__':
    # Create dir for image from user (files) and result from neural network ()
    if not os.path.exists('./result'):
        os.mkdir('result')
    if not os.path.exists('./files'):
        os.mkdir('files')
    # Device for inference NN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Send NN to device
    model.to(device)
    executor.start_polling(dp, skip_updates=True)
