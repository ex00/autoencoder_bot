"""Bot module for autoencoder functionality"""

import logging
import os
from io import BytesIO

from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = os.environ['BOT_KEY']
IMG_FOLDER = os.environ.get('IMG_FOLDER', '/tmp/autoencoder_bot/imgs')

input_images_folder = os.path.join(IMG_FOLDER, 'input')
output_images_folder = os.path.join(IMG_FOLDER, 'output')

for folder in [input_images_folder, output_images_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """This handler will be called when client send `/start` or `/help` commands."""
    await message.reply("Hi!\nI'm Autoencoder Bot!\nPowered by aiogram.\n\nJust send me your photo and I'll make magic!")


@dp.message_handler(content_types=types.ContentType.PHOTO)
async def photo_handler(message: types.Message):
    logging.info('Handle photo file')
    downloaded = await bot.download_file_by_id(message.photo[-1].file_id)

    with BytesIO() as b:
        b.write(downloaded.getvalue())
        target_file = os.path.join(input_images_folder, '{}.jpg'.format(message.chat.id))
        with open(target_file, 'wb') as f:
            f.write(b.getvalue())
        logging.info('photo was save to {}'.format(target_file))
        # b.seek(0)
        # await bot.send_photo(message.chat.id, b)


@dp.message_handler(content_types=types.ContentType.TEXT)
async def echo(message: types.Message):
    await bot.send_message(message.chat.id, message.text)
    # Create media group
    media = types.MediaGroup()
    media.attach_photo('http://lorempixel.com/400/200/cats/', 'Random cat.')
    # # Attach local file
    # media.attach_photo(types.InputFile('data/cat.jpg'), 'Cat!')
    # Done! Send media group
    await message.reply_media_group(media=media)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
