import cv2
from typing import List, Tuple, Any, Optional
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from utils.consts import EYE_AR_THRESH, LEFT_START, LEFT_END, RIGHT_START, RIGHT_END, PREDICTOR, FACE_DETECTOR


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
    """
    Convert a color image to grayscale.

    :param image: A NumPy array representing a color image in RGB format.
    :return: A NumPy array representing the grayscale version of the input image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image


def get_face_rects(image: np.ndarray) -> Optional[List[Tuple]]:
    """
    Detect faces in an image using a face detection model.

    :param image: A NumPy array representing an input image.
    :return: A list of tuples representing detected face rectangles
             as (left, top, right, bottom) coordinates.
    """
    rects = FACE_DETECTOR(image, 0)
    if not rects:
        return None
    largest_face = max(rects, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
    return largest_face


def get_face_shape(gray_image: np.ndarray, largest_face: List[Tuple]) -> np.ndarray:
    """
    Get facial landmarks for the largest detected face in a grayscale image.

    :param gray_image: A NumPy array representing a grayscale image containing a face.
    :param largest_face: A list of tuples representing the coordinates of the largest detected face
                         as (left, top, right, bottom) coordinates.
    :return: A NumPy array containing facial landmark points (x, y) for the largest detected face.
    """
    shape = PREDICTOR(gray_image, largest_face)
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
    return shape[LEFT_START: LEFT_END], shape[RIGHT_START: RIGHT_END]


def are_eyes_blinked(image: np.ndarray) -> (bool, np.ndarray):
    """
    Determine if the eyes in the given image are blinked.

    :param image: A representation of the image containing the person's eyes.
    :return: True if the eyes are blinked, False otherwise.
    """
    eye_coordinates = extract_eyes_coordinates(image)
    if not eye_coordinates:
        return False, None
    left_eye, right_eye = eye_coordinates
    left_eye_ratio = eye_aspect_ratio(left_eye)
    right_eye_ratio = eye_aspect_ratio(right_eye)
    average = (left_eye_ratio + right_eye_ratio) / 2
    return average < EYE_AR_THRESH, eye_coordinates
