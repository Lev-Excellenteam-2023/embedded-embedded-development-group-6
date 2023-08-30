import time
import playsound


def alarm():
    time1 = time.time() + 1
    while time1 > time.time():
        playsound.playsound('Alarm.mp3')


if __name__ == '__main__':
    alarm()