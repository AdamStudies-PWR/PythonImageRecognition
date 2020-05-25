from age_detector import test_image_from_file
from crop_face import crop_faces
import sys
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = sys.argv[1]
# PATH = "/media/adam/DLinux/ripo/Age/train/0-8/9487g0a7.png"

if not os.path.exists(PATH):
    print("Niepoprawna ścieżka! Spróbuj ponownie")
    exit(0)

faces = crop_faces(PATH)
for i in range(len(faces)):
    faces[i] = cv2.resize(faces[i], dsize=(200, 200),
                          interpolation=cv2.INTER_CUBIC)
predictions = []

if(len(faces) >= 1):
    for face in faces:
        predictions.append(test_image_from_file(face))
    plt.figure()
    for i in range(len(faces)):
        ax = plt.subplot(len(faces), 4, i+1)
        im = ax.imshow(faces[i])
        plt.title(predictions[i], fontsize=6)
    plt.tight_layout(pad=30.0)
    plt.show()
    # plt.figure(figsize=(5, 5))
    # plt.imshow(image)

    # # plt.title(OUTPUT[int(output[0][0] < 0)] + " - " + str(output[0][0]))
    # plt.show()

    print(predictions)
    # print(str(output[0][0]))
