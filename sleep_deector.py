import numpy as np
from blinked_detector import are_eyes_blinked
import cv2
import time
from imutils.video import VideoStream
from utils.consts import MAX_BLINKS
import asyncio

cap = cv2.VideoCapture(0)
vs = VideoStream(src=0).start()
font = cv2.FONT_HERSHEY_SIMPLEX


def is_sleeping() -> None:
    counter = 0
    start = time.time()
    while True:
        frame = vs.read()
        key = cv2.waitKey(1)
        if are_eyes_blinked(np.array(frame)) and time.time() - start > 0.5:
            start = time.time()
            counter += 1
        else:
            counter = 0
        cv2.putText(frame, str(counter), (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Surgeon's image", frame)
        # if counter >= MAX_BLINKS:
        #     break
        if key == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()

is_sleeping()
