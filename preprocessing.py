import os
import sys
import cv2 as cv
import shutil as sh
import tensorflow as tf

from crop_face import crop_faces

def double_data(directory, temp_path):
    iter = 0
    for filename in os.listdir(directory):
        faces = crop_faces(directory + "/" + filename)
        temp = "g" + filename.split("g")[1] + "g"
        for face in faces:
            try:
                # cv.imshow(filename, face)
                face = cv.resize(face, dsize=(200, 200), interpolation=cv.INTER_CUBIC)
                cv.imwrite((temp_path + "/" + str(iter) + temp), cv.cvtColor(face, cv.COLOR_RGB2BGR))
                iter = iter + 1
            except:
                print("Błąd openCV!")
        tf.keras.backend.clear_session()
        os.replace(directory + "/" + filename, temp_path + "/" + str(iter) + temp)
        iter = iter + 1


def just_crop(directory, temp_path):
    iter = 0
    for filename in os.listdir(directory):
        faces = crop_faces(directory + "/" + filename)
        temp = "g" + filename.split("g")[1] + "g"
        for face in faces:
            try:
                # cv.imshow(filename, face)
                face = cv.resize(face, dsize=(200, 200), interpolation=cv.INTER_CUBIC)
                cv.imwrite((temp_path + "/" + str(iter) + temp), cv.cvtColor(face, cv.COLOR_RGB2BGR))
                iter = iter + 1
            except:
                print("Błąd openCV!")
        tf.keras.backend.clear_session()

if len(sys.argv) != 2:
    print("Niepoprawnia ilość argumentów!")
    exit(0)

PATH = sys.argv[1]

if not os.path.exists(PATH):
    print("Niepoprawna ścieżka!")
    exit(0)

print("Wybierz tryb: \n [1]. Cropuj twarze i usuń stare \n [2]. Cropuj twarze i zachowaj stare")
ch = input(":> ")

if ch == "1":
    if not os.path.exists(PATH + "/train/male2"):
        os.mkdir(PATH + "/train/male2")
    just_crop(PATH + "/train/male", PATH + "/train/male2")
    sh.rmtree(PATH + "/train/male")
    os.replace(PATH + "/train/male2", PATH + "/train/male")
    
    if not os.path.exists(PATH + "/train/female2"):
        os.mkdir(PATH + "/train/female2")
    just_crop(PATH + "/train/female", PATH + "/train/female2")
    sh.rmtree(PATH + "/train/female")
    os.replace(PATH + "/train/female2", PATH + "/fetrain/male")

    if not os.path.exists(PATH + "/validation/male2"):
        os.mkdir(PATH + "/validation/male2")
    just_crop(PATH + "/validation/male", PATH + "/validation/male2")
    sh.rmtree(PATH + "/validation/male")
    os.replace(PATH + "/validation/male2", PATH + "/validation/male")

    if not os.path.exists(PATH + "/validation/female2"):
        os.mkdir(PATH + "/validation/female2")
    just_crop(PATH + "/validation/female", PATH + "/validation/female2")
    sh.rmtree(PATH + "/validation/female")
    os.replace(PATH + "/validation/female2", PATH + "/fevalidation/male")
elif ch == "2":
    if not os.path.exists(PATH + "/train/male2"):
        os.mkdir(PATH + "/train/male2")
    double_data(PATH + "/train/male", PATH + "/train/male2")
    sh.rmtree(PATH + "/train/male")
    os.replace(PATH + "/train/male2", PATH + "/train/male")

    if not os.path.exists(PATH + "/train/female2"):
        os.mkdir(PATH + "/train/female2")
    double_data(PATH + "/train/female", PATH + "/train/female2")
    sh.rmtree(PATH + "/train/female")
    os.replace(PATH + "/train/female2", PATH + "/fetrain/male")

    if not os.path.exists(PATH + "/validation/male2"):
        os.mkdir(PATH + "/validation/male2")
    double_data(PATH + "/validation/male", PATH + "/validation/male2")
    sh.rmtree(PATH + "/validation/male")
    os.replace(PATH + "/validation/male2", PATH + "/validation/male")

    if not os.path.exists(PATH + "/validation/female2"):
        os.mkdir(PATH + "/validation/female2")
    double_data(PATH + "/validation/female", PATH + "/validation/female2")
    sh.rmtree(PATH + "/validation/female")
    os.replace(PATH + "/validation/female2", PATH + "/fevalidation/male")


