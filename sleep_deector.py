import imutils
import numpy as np
from blinked_detector import are_eyes_blinked
import cv2
import time
from imutils.video import VideoStream
from utils.consts import MAX_BLINKS
from alarms.alarms import alarm
import asyncio

cap = cv2.VideoCapture(0)
vs = VideoStream(src=0).start()
font = cv2.FONT_HERSHEY_SIMPLEX


def is_sleeping() -> None:
    counter = 0
    start = time.time()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        key = cv2.waitKey(1)

        is_blinked, eye_coordinates = are_eyes_blinked(np.array(frame))
        if eye_coordinates:
            leftEyeHull = cv2.convexHull(eye_coordinates[0])
            rightEyeHull = cv2.convexHull(eye_coordinates[1])
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if is_blinked and time.time() - start > 0.5:
            start = time.time()
            counter += 1
        elif not is_blinked:
            counter = 0
        cv2.putText(frame, str(counter), (10, 350), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Surgeon's image", frame)
        if counter >= MAX_BLINKS:
            alarm()
        if key == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()

is_sleeping()
