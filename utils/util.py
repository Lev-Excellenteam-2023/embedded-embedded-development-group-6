from imutils import resize, face_utils
from imutils.video import VideoStream
from numpy import ndarray, array

from cv2 import convexHull, waitKey, drawContours, putText, imshow, destroyAllWindows

from utils.consts import FONT, WIDTH_RESIZE, GREEN, \
    PUT_TEXT_THICKNESS, FONT_SCALE, IMAGE_NAME, X_COORDINATE, Y_COORDINATE

VS = None


def capture() -> (ndarray, int):
    global VS
    if VS is None:
        VS = VideoStream(src=0).start()

    frame = VS.read()
    frame = resize(frame, width=WIDTH_RESIZE)
    key = waitKey(1)
    return frame, key


def close_camera():
    global VS
    if VS is not None:
        destroyAllWindows()
        VS.stop()


def mark_eyes_on_image(frame: ndarray, eye_coordinates: ndarray) -> None:
    left_eye_hull = convexHull(eye_coordinates[0])
    right_eye_hull = convexHull(eye_coordinates[1])
    drawContours(frame, [left_eye_hull], -1, GREEN, PUT_TEXT_THICKNESS)
    drawContours(frame, [right_eye_hull], -1, GREEN, PUT_TEXT_THICKNESS)


def image_show(frame: ndarray, counter: int) -> None:
    putText(frame, str(counter), (X_COORDINATE, Y_COORDINATE), FONT, FONT_SCALE, GREEN, PUT_TEXT_THICKNESS)
    imshow(IMAGE_NAME, frame)


def put_text(frame: ndarray, txt: str):
    putText(frame, txt, (X_COORDINATE, Y_COORDINATE), FONT, FONT_SCALE, GREEN, PUT_TEXT_THICKNESS)
