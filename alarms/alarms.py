from os import getenv
import playsound
from utils.consts import ALARM_PATH, TELEGRAM_MAX
from dotenv import load_dotenv
from requests import post
from numpy import array
from cv2 import imencode
from logging import log, DEBUG, INFO

load_dotenv()

USE_TELEGRAM = getenv('USE_TELEGRAM')
TELEGRAM_TOKEN = getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = getenv('TELEGRAM_CHAT_ID')
API_URL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'

telegram_counter = 0


def alarm(image=None):
    playsound.playsound(ALARM_PATH)

    if USE_TELEGRAM and image is not None:
        send_to_telegram(image)


def send_image_to_telegram(image: array) -> None:
    global telegram_counter

    log(telegram_counter, level=DEBUG)

    if telegram_counter == TELEGRAM_MAX:
        try:
            _, encoded_image = imencode('.jpg', image)

            response = post(API_URL, data={'chat_id': TELEGRAM_CHAT_ID},
                                     files={'photo': ('image.jpg', encoded_image.tobytes())})

            log(response.text, level=INFO)

        except Exception as e:
            log(e, level=DEBUG)
        finally:
            telegram_counter = 0

    telegram_counter += 1


if __name__ == '__main__':
    alarm()
