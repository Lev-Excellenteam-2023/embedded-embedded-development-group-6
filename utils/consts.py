from cv2 import FONT_HERSHEY_SIMPLEX
from os import pardir
from os.path import join, abspath

# paths
PREDICTOR_PATH = abspath(join(__file__, pardir, pardir, 'blink_detector', 'shape_predictor_68_face_landmarks.dat'))
ALARM_PATH = abspath(join(__file__, pardir, pardir, 'alarms', 'Alarm.mp3'))

EYE_AR_THRESH = 0.27
MAX_BLINKS = 2

# image parameters
FONT = FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
WIDTH_RESIZE = 500
PUT_TEXT_THICKNESS = 2
FONT_SCALE = 3
X_COORDINATE = 10
Y_COORDINATE = 350
IMAGE_NAME = "Surgeon's image"

# sleep and eyes blinked detection
EYE_AR_THRESH = 0.22
MAX_BLINKS = 2
UP_SAMPLING = 1

TELEGRAM_MAX = 20
TELEGRAM_MSG = 'SLEEP'
