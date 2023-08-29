import cv2
import os

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
