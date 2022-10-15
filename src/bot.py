from asyncio import sleep
from multiprocessing import Process
from pathlib import Path
from urllib.request import urlretrieve

from aiogram import Bot, Dispatcher, executor, types

from src.datacls import InferenceConfig
from src.inference.consumer import SuperResolutionConsumer


def start_bot(config: InferenceConfig, token: str) -> None:

    input_dir = Path(config.input_dir)
    input_dir.mkdir(exist_ok=True, parents=True)
    target_dir = Path(config.target_dir)
    target_dir.mkdir(exist_ok=True, parents=True)

    # Initialize bot and dispatcher
    bot = Bot(token=token)
    dp = Dispatcher(bot=bot)

    @dp.message_handler(commands=["start"])
    async def process_start_command(message: types.Message):
        await message.answer("Привет! Я бот Tucker, рад что ты присоединился!")

    @dp.message_handler(content_types=["document", "photo"])
    async def scan_message(msg: types.Message):
        try:
            document_id = msg.document.file_id
        except Exception:
            document_id = msg.photo[-1].file_id
        file_info = await bot.get_file(document_id)
        file_id = file_info.file_path

        file_path: str = file_info["file_path"]
        ext = Path(file_path.rsplit("/", maxsplit=1)[-1]).suffix
        file_name = str(msg.from_user.id) + ext
        urlretrieve(
            f"https://api.telegram.org/file/bot{token}/{file_id}",
            str(input_dir / file_name),
        )
        await msg.reply("Файл успешно сохранён и нейросеть начала его обрабатывать")
        # Preprocess image with NN
        completed_file = target_dir / file_name
        while True:
            if completed_file.is_file():
                break
            await sleep(1)
        with open(completed_file, "rb") as photo:
            await bot.send_document(msg.from_user.id, photo)
            completed_file.unlink()

    consumer = SuperResolutionConsumer(config=config)
    producer = executor.start_polling

    consumer_process = Process(target=consumer.run)
    bot_process = Process(target=producer, args=(dp,), kwargs={"skip_updates": True})
    processes = [consumer_process, bot_process]

    for process in processes:
        process.start()
    for process in processes:
        process.join()
