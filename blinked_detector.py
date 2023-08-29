import cv2
import os
from typing import List, Tuple, Any
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from utils.consts import PREDICTOR_PATH, EYE_AR_THRESH

cascades_dir = os.path.join(os.getcwd(), 'cascades')
face_Cascade_dir = os.path.join(cascades_dir, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_Cascade_dir)
left_start, left_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
right_start, right_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


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


def extract_eyes_coordinates(face_image: np.ndarray) -> Tuple[Any, Any]:
    """
    Extracts the coordinates of the left and right eyes from a face image.

    :param face_image: A NumPy array representing a face image.
    :return: Two NumPy arrays containing the coordinates of the left and right eyes.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    face_image = imutils.resize(face_image, width=500)
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray_image, 1)
    shape = predictor(gray_image, rects[0])
    shape = face_utils.shape_to_np(shape)
    return shape[left_start: left_end], shape[right_start: right_end]


def are_eyes_blinked(image: np.ndarray) -> bool:
    """
    Determine if the eyes in the given image are blinked.

    :param image: A representation of the image containing the person's eyes.
    :return: True if the eyes are blinked, False otherwise.
    """
    left_eye, right_eye = extract_eyes_coordinates(image)
    left_eye_ratio = eye_aspect_ratio(left_eye)
    right_eye_ratio = eye_aspect_ratio(right_eye)
    average = (left_eye_ratio + right_eye_ratio) / 2
    return average < EYE_AR_THRESH
