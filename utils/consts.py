from imutils import face_utils
import dlib
import cv2
from imutils.video import VideoStream
import os

CAP = cv2.VideoCapture(0)
VS = VideoStream(src=0).start()
FONT = cv2.FONT_HERSHEY_SIMPLEX

PREDICTOR_PATH = 'blink_detector\\shape_predictor_68_face_landmarks.dat'
LEFT_START, LEFT_END = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
RIGHT_START, RIGHT_END = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)
FACE_DETECTOR = dlib.get_frontal_face_detector()

EYE_AR_THRESH = 0.22
MAX_BLINKS = 2


ALARM_PATH = os.path.join(__file__, os.pardir, os.pardir, 'alarms', 'Alarm.mp3')