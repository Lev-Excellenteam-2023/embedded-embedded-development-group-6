import cv2
import os
from typing import List, Tuple, Any, Optional
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from utils.consts import PREDICTOR_PATH, EYE_AR_THRESH
import time

# cascades_dir = os.path.join(os.getcwd(), 'cascades')
# face_Cascade_dir = os.path.join(cascades_dir, 'haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier(face_Cascade_dir)
left_start, left_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
right_start, right_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_detector = dlib.get_frontal_face_detector()


def eye_aspect_ratio(eye_landmarks: List[Tuple]) -> float:
    """
    Calculate the eye aspect ratio (EAR) to detect eye blinks and eye openness.

    The EAR is computed as the average of two ratios:
    1. The distance between the vertical eye landmarks (upper and lower eyelids) divided by the horizontal distance
       between the horizontal eye landmarks (the inner and outer corners of the eye).

    :param eye_landmarks: A list of tuples containing the (x, y) coordinates of six eye landmarks in the following order:
                          [left_eye_corner, right_eye_corner, upper_eye_lid, lower_eye_lid, inner_eye_corner, outer_eye_corner]
    :return: The computed eye aspect ratio (EAR).
    """
    left_width = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    right_width = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    length = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

    ratio = (left_width + right_width) / (2.0 * length)
    return ratio


def get_gray_image(image: np.ndarray) -> np.ndarray:
    image = imutils.resize(image, width=500)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image


def get_face_rects(image: np.ndarray) -> Optional[List[Tuple]]:
    rects = face_detector(image, 0)
    if not rects:
        return None
    largest_face = max(rects, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
    return largest_face

def get_face_shape(gray_image: np.ndarray, largest_face: List[Tuple]) -> np.ndarray:
    shape = predictor(gray_image, largest_face)
    shape = face_utils.shape_to_np(shape)
    return shape


def extract_eyes_coordinates(image: np.ndarray) -> Optional[Tuple[Any, Any]]:
    """
    Extracts the coordinates of the left and right eyes from a face image.

    :param image: A NumPy array representing a face image.
    :return: Two NumPy arrays containing the coordinates of the left and right eyes.
    """
    gray_image = get_gray_image(image)
    rect = get_face_rects(gray_image)
    if not rect:
        return None
    shape = get_face_shape(gray_image, rect)
    return shape[left_start: left_end], shape[right_start: right_end]


def are_eyes_blinked(image: np.ndarray) -> bool:
    """
    Determine if the eyes in the given image are blinked.

    :param image: A representation of the image containing the person's eyes.
    :return: True if the eyes are blinked, False otherwise.
    """
    eye_coordinates = extract_eyes_coordinates(image)
    if not eye_coordinates:
        return False
    left_eye, right_eye = eye_coordinates
    left_eye_ratio = eye_aspect_ratio(left_eye)
    right_eye_ratio = eye_aspect_ratio(right_eye)
    average = (left_eye_ratio + right_eye_ratio) / 2
    return average < EYE_AR_THRESH


if __name__ == '__main__':
    # frame = cv2.imread(('images\\20.png'))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plt.imshow(frame)
    # plt.show()
    print(are_eyes_blinked(cv2.imread('images\\18.png')))
