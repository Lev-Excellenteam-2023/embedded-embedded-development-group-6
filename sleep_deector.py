from imutils import resize, face_utils
from imutils.video import VideoStream
from numpy import ndarray,  array
from blinked_detector import are_eyes_blinked
from cv2 import convexHull, waitKey, drawContours, putText, imshow, destroyAllWindows
from time import time
from utils.consts import MAX_BLINKS, FONT, WIDTH_RESIZE, GREEN,\
    PUT_TEXT_THICKNESS, FONT_SCALE, IMAGE_NAME, X_COORDINATE, Y_COORDINATE
from alarms.alarms import alarm


VS = VideoStream(src=0).start()



def capture() -> (ndarray, int):
    frame = VS.read()
    frame = resize(frame, width=WIDTH_RESIZE)
    key = waitKey(1)
    return frame, key


def mark_eyes_on_image(frame: ndarray, eye_coordinates: ndarray) -> None:
    left_eye_hull = convexHull(eye_coordinates[0])
    right_eye_hull = convexHull(eye_coordinates[1])
    drawContours(frame, [left_eye_hull], -1, GREEN, PUT_TEXT_THICKNESS)
    drawContours(frame, [right_eye_hull], -1, GREEN, PUT_TEXT_THICKNESS)


def handle_counter(counter: int, is_blinked: bool, start_time: float) -> (int, float):
    if is_blinked and time() - start_time > 0.5:
        return counter + 1, time()
    elif not is_blinked:
        return 0, start_time
    else:
        return counter, start_time


def image_show(frame: ndarray, counter: int) -> None:
    putText(frame, str(counter), (X_COORDINATE, Y_COORDINATE), FONT, FONT_SCALE, GREEN, PUT_TEXT_THICKNESS)
    imshow(IMAGE_NAME, frame)


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
