from genders import use_model_images
from crop_face import crop_faces
import sys
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = sys.argv[1]
# PATH = "/run/media/adam/DLinux/BackupRIPO/14g1a61.png"

OUTPUT = ["Chłopczyk", "Dziewczynka"]

if not os.path.exists(PATH):
    print("Niepoprawna ścieżka! Spróbuj ponownie")
    exit(0)

faces = crop_faces(PATH)
for i in range(len(faces)):
    faces[i] = cv2.resize(faces[i], dsize=(200, 200),
                          interpolation=cv2.INTER_CUBIC)

if(len(faces) >= 1):
    predictions = use_model_images(faces)
    plt.figure()
    for i in range(len(faces)):
        ax = plt.subplot(len(faces), 4, i+1)
        im = ax.imshow(faces[i])
        plt.title(OUTPUT[int(predictions[i][0] < 0)] +
                  " - " + str(predictions[i][0]), fontsize=6)
    plt.tight_layout(pad=30.0)
    plt.show()
    # plt.figure(figsize=(5, 5))
    # plt.imshow(image)

    # # plt.title(OUTPUT[int(output[0][0] < 0)] + " - " + str(output[0][0]))
    # plt.show()

    print(predictions)
    # print(str(output[0][0]))
