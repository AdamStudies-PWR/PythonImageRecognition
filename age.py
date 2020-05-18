import pandas as pd
import keras
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


# MALES
MDIR = "/media/adam/DLinux/ripo/Croped/train/male/"
# FEMALES
FDIR = "/media/adam/DLinux/ripo/Croped/train/female/"

# DATA
target_size = (224, 224)


def processFiles(name):
    temp = name.split("a")
    age = temp[1].split(".")
    return age[0]


def getImagePixels(image_path):
    img = image.load_img(image_path, grayscale=False, target_size=target_size)
    result = image.img_to_array(img).reshape(1, -1)[0]
    return result


age = []
full_path = []

for filename in os.listdir(MDIR):
    temp = processFiles(filename)
    if 0 < int(temp) <= 100:
        age.append(temp)
        full_path.append(MDIR + filename)

for filename in os.listdir(FDIR):
    temp = processFiles(filename)
    if 0 < int(temp) <= 100:
        age.append(temp)
        full_path.append(FDIR + filename)

instances = len(age)
data = pd.DataFrame({'age': age, 'full_path': full_path}, index = range(0, instances))
data['pixels'] = data['full_path'].apply(getImagePixels)


classes = 101
target = data['age'].values
target_classes = keras.utils.to_categorical(target, classes)
features = []
for i in range(0, instances):
    features.append(data['pixels'].values[i])
print("Crash Test2!")
#features = np.array(features)
print("Crash Test3!")
#features = features.reshape(features.shape[0], 224, 224, 3)

train_x, test_x, train_y, test_y = train_test_split(features, target_classes, test_size=0.30)
