import time
import webbrowser
from utils.consts import ALARM_PATH


def alarm():
    time1 = time.time() + 1
    while time1 > time.time():
        webbrowser.open(ALARM_PATH)


if __name__ == '__main__':
    alarm()
