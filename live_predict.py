import cv2 as cv
import tensorflow as tf

from crop_face import locate_faces
from genders import use_model_images

cv.namedWindow("Gender and Age detection system PRO")

capture = cv.VideoCapture(0)

if capture.isOpened():
    active, frame = capture.read()

counter = 0
while active:
    if counter == 0:
        cords, faces = locate_faces(frame)
        # predictions = use_model_images(faces)
        tf.keras.backend.clear_session()
    for box in cords:
        cv.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255),  4)
    cv.imshow("Gender and Age detection system PRO", frame)
    key = cv.waitKey(10)
    if key == 27:
        break;
    counter = counter + 1
    if counter == 10:
        counter = 0
    active, frame = capture.read()

cv.destroyWindow("Gender and Age detection system PRO")