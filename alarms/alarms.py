import time
import playsound
from utils.consts import ALARM_PATH


def alarm():
    time1 = time.time() + 1
    while time1 > time.time():
        playsound.playsound(ALARM_PATH)


if __name__ == '__main__':
    alarm()
