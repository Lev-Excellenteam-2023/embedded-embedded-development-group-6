import cv2
import os

cascades_dir = os.path.join(os.getcwd(), 'cascades')
face_Cascade_dir = os.path.join(cascades_dir, 'haarcascade_frontalface_default.xml')
eye_Cascade_dir = os.path.join(cascades_dir, 'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(face_Cascade_dir)
eye_cascade = cv2.CascadeClassifier(eye_Cascade_dir)


def detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    return face_rect


def detect_eyes_in_face_img(face_img):
    face_img = face_img.copy()
    eyes_rect = eye_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=9)
    return eyes_rect


def eye_detector(img):
    resize_img = cv2.resize(img, (1000, 600))

    faces_rect = detect_face(resize_img)

    big_face_rect = max(faces_rect, key=lambda f: f[2])

    x, y, w, h = big_face_rect

    face_img = resize_img[y:y + h, x:x + w, :]

    eyes_rects = detect_eyes_in_face_img(face_img)
    show_img(resize_img)
    show_img(face_img)


    for (x, y, w, h) in eyes_rects:
       cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 0, 0), 10)

    show_img(face_img)


   # eyes_images = [ face_img[y:y + h, x:x + w, :] for x, y, w, h in eyes_rects]

  #  return eyes_images




def show_img(img):
    while True:
        cv2.imshow('', img)
        code = cv2.waitKey(10)
        if code == ord('q'):
            break


if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\dov31\Desktop\smash\IMG_0153.JPG')

    eye_detector(img)

