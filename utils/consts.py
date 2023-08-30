from cv2 import FONT_HERSHEY_SIMPLEX
from os import pardir
from os.path import join

FONT = FONT_HERSHEY_SIMPLEX

PREDICTOR_PATH = join(__file__, pardir, pardir, 'blink_detector', 'shape_predictor_68_face_landmarks.dat')

EYE_AR_THRESH = 0.22
MAX_BLINKS = 2

ALARM_PATH = join(__file__, pardir, pardir, 'alarms', 'Alarm.mp3')

GREEN = (0, 255, 0)
WIDTH_RESIZE = 500
PUT_TEXT_THICKNESS = 2
FONT_SCALE = 3
X_COORDINATE = 10
Y_COORDINATE = 350

IMAGE_NAME = "Surgeon's image"

UP_SAMPLING = 1
