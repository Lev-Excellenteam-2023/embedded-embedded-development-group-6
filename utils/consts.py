from imutils import face_utils
from dlib import shape_predictor, get_frontal_face_detector
from cv2 import VideoCapture, FONT_HERSHEY_SIMPLEX
from imutils.video import VideoStream
from os import pardir
from os.path import join


VS = VideoStream(src=0).start()
FONT = FONT_HERSHEY_SIMPLEX

PREDICTOR_PATH = join(__file__, pardir, pardir, 'blink_detector', 'shape_predictor_68_face_landmarks.dat')
LEFT_START, LEFT_END = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
RIGHT_START, RIGHT_END = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
PREDICTOR = shape_predictor(PREDICTOR_PATH)
FACE_DETECTOR = get_frontal_face_detector()

EYE_AR_THRESH = 0.22
MAX_BLINKS = 2


ALARM_PATH = join(__file__, pardir, pardir, 'alarms', 'Alarm.mp3')