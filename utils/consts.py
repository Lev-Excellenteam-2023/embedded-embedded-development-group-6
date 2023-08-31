from cv2 import FONT_HERSHEY_SIMPLEX
from os import pardir
from os.path import join, abspath

# paths
PREDICTOR_PATH = abspath(join(__file__, pardir, pardir, 'blink_detector', 'shape_predictor_68_face_landmarks.dat'))
ALARM_PATH = abspath(join(__file__, pardir, pardir, 'alarms', 'Alarm.mp3'))

# image parameters
FONT = FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WIDTH_RESIZE = 500
PUT_TEXT_THICKNESS = 2
FONT_SCALE = 3
X_COORDINATE = 10
Y_COORDINATE = 350
IMAGE_NAME = "Surgeon's image"

# sleep and eyes blinked detection
EYE_AR_THRESH = 0.22
MAX_BLINKS = 3
UP_SAMPLING = 1
PERIOD_TIME = 0.5

TELEGRAM_MAX = 2
TELEGRAM_MSG = 'SLEEP'
