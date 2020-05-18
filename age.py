import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
import os


# MALES
MDIR = "/media/adam/DLinux/ripo/Croped/train/male/"
# FEMALES
FDIR = "/media/adam/DLinux/ripo/Croped/train/female/"

#PRE DONE WEIGHTS
PWEIGHTS = "/home/adam/PWR/RIPO/PythonImageRecognition/vgg_face_weights.h5"

#SAVEPOINTS
CHKPOINT = "/home/adam/PWR/RIPO/PythonImageRecognition/agecheck/age_model.hdf5"

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

# Nie używam nic z kobiet bo w przeciwnym wypadku zjada mi cały RAM
# for filename in os.listdir(FDIR):
#     temp = processFiles(filename)
#     if 0 < int(temp) <= 100:
#         age.append(temp)
#         full_path.append(FDIR + filename)

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

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights(PWEIGHTS)

for layer in model.layers[:-7]:
    layer.trainable = False


base_model_output = Sequential()
base_model_output = Conv2D(101, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)

age_model = tf.keras.Model(inputs=model.input, outputs=base_model_output)

age_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=''
                               , monitor="val_loss", verbose=1, save_best_only=True, mode='auto')

epochs = 250;
batch_size = 256

for i in range(epochs):
    print("epoch ", i)

ix_train = np.random.choice(train_x.shape[0], size=batch_size)

score = age_model.fit(train_x[ix_train], train_y[ix_train]
                      , epochs=1, validation_data=(test_x, test_y), callbacks=[checkpointer])

