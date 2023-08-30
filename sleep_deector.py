from imutils import resize
from numpy import ndarray,  array
from blinked_detector import are_eyes_blinked
from cv2 import convexHull, waitKey, drawContours, LINE_AA, putText, imshow, destroyAllWindows
from time import time
from utils.consts import MAX_BLINKS, VS, FONT
from alarms.alarms import alarm


def capture() -> (ndarray, int):
    frame = VS.read()
    frame = resize(frame, width=500)
    key = waitKey(1)
    return frame, key


def mark_eyes_on_image(frame: ndarray, eye_coordinates: ndarray) -> None:
    left_eye_hull = convexHull(eye_coordinates[0])
    right_eye_hull = convexHull(eye_coordinates[1])
    drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
    drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)


def handle_counter(counter: int, is_blinked: bool, start_time: float) -> (int, float):
    if is_blinked and time() - start_time > 0.5:
        return counter + 1, time()
    elif not is_blinked:
        return 0, start_time
    else:
        return counter, start_time


def image_show(frame: ndarray, counter: int) -> None:
    putText(frame, str(counter), (10, 350), FONT, 3, (0, 255, 0), 2, LINE_AA)
    imshow("Surgeon's image", frame)


def is_sleeping() -> None:
    counter = 0
    start = time()
    while True:
        frame, key = capture()
        is_blinked, eye_coordinates = are_eyes_blinked(array(frame))
        if eye_coordinates:
            mark_eyes_on_image(frame, eye_coordinates)
        counter, start = handle_counter(counter, is_blinked, start)
        image_show(frame, counter)
        if counter >= MAX_BLINKS:
            alarm()
        if key == ord('q'):
            break
    destroyAllWindows()
    VS.stop()


if __name__ == '__main__':
    is_sleeping()
