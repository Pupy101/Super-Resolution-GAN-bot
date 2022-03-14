import argparse
import os

from asyncio import sleep
from multiprocessing import Process
from pathlib import Path
from urllib.request import urlretrieve

import torch

from aiogram import Bot, Dispatcher, executor, types

from src.inference.consumer import SuperResolutionConsumer
from src.model import SuperResolutionGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--api_token", dest="token", help="Telegram bot api token"
    )
    parser.add_argument("-i", "--input", help="directory for download image for bot")
    parser.add_argument("-t", "--target", help="directory preprocessed images from bot")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        dest="batch",
        help="Batch size for consumer with super resolution network",
    )
    parser.add_argument("-w", "--weight", help="path to model weight")
    args = parser.parse_args()

    os.makedirs(args.input, exist_ok=True)
    os.makedirs(args.target, exist_ok=True)

    input_dir = Path(args.input)
    target_dir = Path(args.target)

    # bot setup
    # Set button
    button_hi = types.KeyboardButton("Привет! 👋\nРасскажи о себе")
    greet_kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    greet_kb.add(button_hi)

    # Initialize bot and dispatcher
    bot = Bot(token=args.token)
    dp = Dispatcher(bot)

    @dp.message_handler(commands=["start"])
    async def process_start_command(message: types.Message):
        global state, trigger
        state = "start"
        trigger = True
        await message.answer(
            "Привет! Я бот Tucker, рад что ты присоединился!", reply_markup=greet_kb
        )

    @dp.message_handler(content_types=["document", "photo"])
    async def scan_message(msg: types.Message):
        try:
            document_id = msg.document.file_id
        except Exception:
            document_id = msg.photo[-1].file_id
        file_info = await bot.get_file(document_id)
        fi = file_info.file_path
        _, ext = os.path.splitext(file_info["file_path"].split("/")[-1])
        urlretrieve(
            f"https://api.telegram.org/file/bot{args.token}/{fi}",
            str(input_dir / f"{str(msg.from_user.id)+ext}"),
        )
        await bot.send_message(
            msg.from_user.id,
            "Файл успешно сохранён и нейросеть начала его обрабатывать",
        )
        # Preprocess image with NN
        completed_file = target_dir / f"{str(msg.from_user.id)+ext}"
        while True:
            if completed_file.is_file():
                break
            await sleep(1)
        with open(completed_file, "rb") as photo:
            await bot.send_document(msg.from_user.id, photo)
            completed_file.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperResolutionGenerator()

    consumer = SuperResolutionConsumer(
        model=model,
        weight=args.weight,
        device=device,
        batch_size=args.batch,
        input_dir=args.input,
        target_dir=args.target,
    )
    producer = executor.start_polling

    process = []
    process.append(Process(target=consumer.run))
    process.append(Process(target=producer, args=(dp,), kwargs={"skip_updates": True}))

    [p.join() for p in process]
