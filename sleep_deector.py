import imutils
import numpy as np
from blinked_detector import are_eyes_blinked
import cv2
import time
from utils.consts import MAX_BLINKS, VS, FONT, CAP
from alarms.alarms import alarm


def capture() -> (np.ndarray, int):
    frame = VS.read()
    frame = imutils.resize(frame, width=500)
    key = cv2.waitKey(1)
    return frame, key


def mark_eyes_on_image(frame: np.ndarray, eye_coordinates: np.ndarray) -> None:
    left_eye_hull = cv2.convexHull(eye_coordinates[0])
    right_eye_hull = cv2.convexHull(eye_coordinates[1])
    cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)


def handle_counter(counter: int, is_blinked: bool, start_time: float) -> (int, float):
    if is_blinked and time.time() - start_time > 0.5:
        return counter + 1, time.time()
    elif not is_blinked:
        return 0, start_time
    else:
        return counter, start_time


def image_show(frame: np.ndarray, counter: int) -> None:
    cv2.putText(frame, str(counter), (10, 350), FONT, 3, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Surgeon's image", frame)


def is_sleeping() -> None:
    counter = 0
    start = time.time()
    while True:
        frame, key = capture()
        is_blinked, eye_coordinates = are_eyes_blinked(np.array(frame))
        if eye_coordinates:
            mark_eyes_on_image(frame, eye_coordinates)
        counter, start = handle_counter(counter, is_blinked, start)
        image_show(frame, counter)
        if counter >= MAX_BLINKS:
            alarm()
        if key == ord('q'):
            break
    CAP.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    is_sleeping()
