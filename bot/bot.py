import asyncio
import time
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command

from model_utils import predict_image_from_bytes
from config import API_TOKEN

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

@dp.message(Command('start'))
async def cmd_start(message: Message):
    await message.answer("Привет! Отправь фото фрукта/овоща — я скажу, здоровый он или гнилой!")

@dp.message(lambda m: m.photo)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    file_id = photo.file_id
    file = await bot.get_file(file_id)
    file_bytes = await bot.download_file(file.file_path)
    img_bytes = file_bytes.read()

    start = time.time()
    label, confidence = predict_image_from_bytes(img_bytes)
    elapsed = time.time() - start

    threshold = 0.7

    if confidence < threshold:
        answer = (
            f"Внимание!\n"
            f"Модель не уверена в предсказании (уверенность {confidence:.2f}).\n"
            f"Возможно, на фото не фрукт/овощ или такого класса нет в обучении.\n\n"
            f"Модель думает, что это: <b>{label}</b>\n"
            f"Время инференса: <b>{elapsed*1000:.1f} мс</b>\n\n"
            f"Можешь отправить другое фото для анализа."
        )
    else:
        answer = (
            f"Модель думает, что это:\n<b>{label}</b>\n"
            f"Уверенность: <b>{confidence*100:.1f}%</b>\n"
            f"Время инференса: <b>{elapsed*1000:.1f} мс</b>\n\n"
            f"Можешь отправить другое фото для анализа."
        )

    await message.reply(answer, parse_mode="HTML")

@dp.message()
async def handle_any(message: Message):
    await message.reply("Пожалуйста, отправь фото фрукта или овоща!")

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
