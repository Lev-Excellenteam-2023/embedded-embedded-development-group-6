import cv2
import os
from scipy.spatial import distance as dist
import dlib



def eye_aspect_ratio(eye):
    """
    Calculate the eye aspect ratio (EAR) to detect eye blinks and eye openness.

    The EAR is computed as the average of two ratios:
    1. The distance between the vertical eye landmarks (upper and lower eyelids) divided by the horizontal distance
       between the horizontal eye landmarks (the inner and outer corners of the eye).

    :param eye_landmarks: A list of tuples containing the (x, y) coordinates of six eye landmarks in the following order:
                          [left_eye_corner, right_eye_corner, upper_eye_lid, lower_eye_lid, inner_eye_corner, outer_eye_corner]
    :return: The computed eye aspect ratio (EAR).
    """

    left_width = dist.euclidean(eye[1], eye[5])
    right_width = dist.euclidean(eye[2], eye[4])

    length = dist.euclidean(eye[0], eye[3])

    ratio = (left_width + right_width) / (2.0 * length)

    return ratio



cascades_dir = os.path.join(os.getcwd(), 'cascades')
face_Cascade_dir = os.path.join(cascades_dir, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_Cascade_dir)


def detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    return face_rect


def show_img(img):
    while True:
        cv2.imshow('', img)
        code = cv2.waitKey(10)
        if code == ord('q'):
            break


if __name__ == '__main__':
    img = cv2.imread(r'your image')
