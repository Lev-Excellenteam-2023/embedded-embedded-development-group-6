from time import time

import utils.util as util
from utils.consts import MAX_BLINKS

from alarms.alarms import alarm
from blinked_detector import are_eyes_blinked

from numpy import array


def handle_counter(counter: int, is_blinked: bool, start_time: float) -> (int, float):
    if is_blinked and time() - start_time > 0.5:
        return counter + 1, time()
    elif not is_blinked:
        return 0, start_time
    else:
        return counter, start_time


def is_sleeping() -> None:
    counter = 0
    start = time()
    while True:
        frame, key = util.capture()
        is_blinked, eye_coordinates = are_eyes_blinked(array(frame))
        if eye_coordinates:
            util.mark_eyes_on_image(frame, eye_coordinates)
        counter, start = handle_counter(counter, is_blinked, start)
        util.image_show(frame, counter)
        if counter >= MAX_BLINKS:
            alarm(frame)
        if key == ord('q'):
            break

    util.close_camera()


if __name__ == '__main__':
    is_sleeping()
