from utils.util import capture, mark_eyes_on_image, close_camera,   put_text, image_show
from blinked_detector import are_eyes_blinked
from numpy import array
from cv2 import hconcat, imshow

COUNT_WIN = 20


def split_img(frame :array):
    h, w, channels = frame.shape
    half = w // 2
    left_part = frame[:, :half]
    right_part = frame[:, half:]
    return left_part, right_part

def play():
    counter_left = 0
    counter_right = 0

    right_prev_blink = False
    left_prev_blink = False

    while True:
        frame, key = capture()
        left_part, right_part = split_img(frame)

        is_blinked_left, eye_coordinates_left = are_eyes_blinked(array(left_part))
        is_blinked_right, eye_coordinates_right = are_eyes_blinked(array(right_part))

        if eye_coordinates_left:
            mark_eyes_on_image(left_part, eye_coordinates_left)

        if eye_coordinates_right:
            mark_eyes_on_image(right_part, eye_coordinates_right)

        if is_blinked_left and not left_prev_blink:
            left_prev_blink = True
            counter_left += 1
        elif eye_coordinates_left:
            left_prev_blink = False

        if is_blinked_right and not right_prev_blink:
            right_prev_blink = True
            counter_right += 1
        elif eye_coordinates_right:
            right_prev_blink = False

        put_text(left_part, str(counter_left))
        put_text(right_part, str(counter_right))

        tow_players_image = hconcat([right_part, left_part])
        imshow('game', tow_players_image)

        if counter_right >= COUNT_WIN:
            winner = 'right'
            break

        if counter_left >= COUNT_WIN:
            winner = 'left'
            break


    close_camera()

if __name__ == '__main__':
  play()