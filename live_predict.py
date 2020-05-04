import cv2 as cv
import tensorflow as tf

from crop_face import locate_faces, init_detector
from genders import use_model_images, init_model

cv.namedWindow("Gender and Age detection system PRO")

OUTPUT = ["Chłopczyk", "Dziewczynka"]

detector = init_detector()

capture = cv.VideoCapture(0)

if capture.isOpened():
    active, frame = capture.read()
(image_generator, model) = init_model()
counter = 0
while active:
    if counter == 0:
        cords, faces = locate_faces(frame, detector)
        for i in range(len(faces)):
            faces[i] = cv.resize(faces[i], dsize=(
                200, 200), interpolation=cv.INTER_CUBIC)
        if len(faces) >= 1:
            predictions = use_model_images(faces, image_generator, model)
        tf.keras.backend.clear_session()
    iter = 0
    for box in cords:
        cv.rectangle(frame, (box[0], box[1]), (box[0] +
                                               box[2], box[1] + box[3]), (0, 0, 255),  4)
        cv.putText(frame, (OUTPUT[int(predictions[iter][0] < 0)] + " - " + str(predictions[iter][0])),
                   (box[0] - 15, box[1] + box[3] + 15), cv.FONT_ITALIC, 0.5, (0, 0, 255), 2, cv.LINE_4)
        iter = iter + 1
    cv.imshow("Gender and Age detection system PRO", frame)
    key = cv.waitKey(10)
    if key == 27:
        break
    counter = counter + 1
    if counter == 10:
        counter = 0
    active, frame = capture.read()

cv.destroyWindow("Gender and Age detection system PRO")
