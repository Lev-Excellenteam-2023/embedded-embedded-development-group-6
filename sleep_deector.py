from typing import Tuple

from imutils import resize, face_utils
from imutils.video import VideoStream
from numpy import ndarray, array
from blinked_detector import are_eyes_blinked
from cv2 import convexHull, waitKey, drawContours, putText, imshow, destroyAllWindows
from time import time
from utils.consts import MAX_BLINKS, FONT, WIDTH_RESIZE, GREEN, PERIOD_TIME, \
    PUT_TEXT_THICKNESS, FONT_SCALE, IMAGE_NAME, X_COORDINATE, Y_COORDINATE, RED
from alarms.alarms import alarm

VS = VideoStream(src=0).start()


def capture() -> (ndarray, int):
    """
    Captures a frame from a video source, resizes it, and waits for a key press.

    :return: A tuple containing the captured frame (as a NumPy ndarray) and the key code.
    """
    frame = VS.read()
    frame = resize(frame, width=WIDTH_RESIZE)
    key = waitKey(1)
    return frame, key


def mark_eyes_on_image(frame: ndarray, eye_coordinates: ndarray, text_color: Tuple[int, int, int]) -> None:
    """
    Marks the eyes on an image frame using convex hulls.

    :param text_color: text_color: The color the eyes should be colored.
    :param frame: The image frame to mark the eyes on.
    :param eye_coordinates: A NumPy array containing coordinates of the eyes.
    :return: None

    """
    left_eye_hull = convexHull(eye_coordinates[0])
    right_eye_hull = convexHull(eye_coordinates[1])
    drawContours(frame, [left_eye_hull], -1, text_color, PUT_TEXT_THICKNESS)
    drawContours(frame, [right_eye_hull], -1, text_color, PUT_TEXT_THICKNESS)


def handle_counter(counter: int, is_blinked: bool, start_time: float, frame) -> (int, float, bool):
    """
    Update a counter based on whether an eye blink is detected and a timing threshold.

    :param frame: The frame for sending to alarm.
    :param counter: An integer representing the current count.
    :param is_blinked: A boolean indicating whether an eye
    blink is detected.
    :param start_time: A float representing the start time of the measurement.
    :return: A tuple containing the updated counter (int) and the updated start time (float) and a flag that says if half a second has
    passed (bool).
    """
    if is_blinked and time() - start_time > PERIOD_TIME:
        counter += 1
        if counter >= MAX_BLINKS:
            alarm(frame)
        start_time = time()
    elif not is_blinked:
        counter = 0
    return counter, start_time


def image_show(frame: ndarray, counter: int, text_color: Tuple[int, int, int]) -> None:
    """
    Display an image frame with a counter value overlaid.

    :param text_color: The color the number should be colored.
    :param frame: A numpy.ndarray representing the image frame to display.
    :param counter: An integer representing the counter value to display on the image.
    :return: None
    """
    putText(frame, str(counter), (X_COORDINATE, Y_COORDINATE), FONT, FONT_SCALE, text_color, PUT_TEXT_THICKNESS)
    imshow(IMAGE_NAME, frame)


def is_sleeping() -> None:
    """
    Monitor for signs of drowsiness (eye blinks) and trigger an alarm if necessary.

    :return: None
    """
    counter = 0
    start = time()
    while True:
        frame, key = capture()
        is_blinked, eye_coordinates = are_eyes_blinked(array(frame))
        text_color = RED if is_blinked else GREEN
        if eye_coordinates:
            mark_eyes_on_image(frame, eye_coordinates, text_color)
        counter, start = handle_counter(counter, is_blinked, start, frame)
        image_show(frame, counter, text_color)

        if key == ord('q'):
            break
    destroyAllWindows()
    VS.stop()


if __name__ == '__main__':
    is_sleeping()
