import cv2 as cv

cv.namedWindow("Gender and Age detection system PRO")

capture = cv.VideoCapture(0)

if capture.isOpened():
    active, frame = capture.read()

while active:
    cv.imshow("Gender and Age detection system PRO", frame)

    # DO SOMETHING HERE

    key = cv.waitKey(20)
    if key == 27:
        break;
    active, frame = capture.read()

cv.destroyWindow("Gender and Age detection system PRO")