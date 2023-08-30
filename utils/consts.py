from imutils import face_utils
import dlib

PREDICTOR_PATH = 'blink_detector\\shape_predictor_68_face_landmarks.dat'
LEFT_START, LEFT_END = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
RIGHT_START, RIGHT_END = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)
FACE_DETECTOR = dlib.get_frontal_face_detector()

EYE_AR_THRESH = 0.22
MAX_BLINKS = 2

ALARM_PATH = r'alarms\Alarm.mp3'  # os.path.join(os.getcwd(), os.pardir, 'alarms','Alarm.mp3')
