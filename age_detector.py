import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import datetime
import tensorflow as tf

img_width, img_height = 200, 200
top_model_weights_path = "/home/adam/PWR/RIPO/PythonImageRecognition/model/bottleneck_fc_model.h5"
btneck_file = "/home/adam/PWR/RIPO/PythonImageRecognition/model/bottleneck_features_train.npy"

train_data_dir = "/media/adam/DLinux/ripo/Age/train"
validation_data_dir = "/media/adam/DLinux/ripo/Age/validation/"

SAVE_PATH = '/home/adam/PWR/RIPO/PythonImageRecognition/saved_model/model'

epochs = 8

batch_size = 50

vgg16 = applications.VGG16(include_top=False, weights="imagenet")
datagen = ImageDataGenerator(rescale=1. / 255)

def load_model():
    # model = Sequential()
    # model.load_weights(top_model_weights_path)
    # return model
    return tf.keras.models.load_model(top_model_weights_path)


def read_image(file_path):
   image = load_img(file_path, target_size=(200, 200))
   image = img_to_array(image)
   image = np.expand_dims(image, axis=0)
   image /= 255.
   return image


def test_single_image(path, model):
  age_gaps = ["[0 - 8]", "[9 - 14]", "[15 - 23]", "[24 - 35]", "[36 - 45]", "[46 - 54]", "[55 - 65]", "[66+"]
  images = read_image(path)
  bt_prediction = vgg16.predict(images)
  preds = model.predict_proba(bt_prediction)
  return preds


def test_image_from_file(image):
  age_gaps = ["[0 - 8]", "[9 - 14]", "[15 - 23]", "[24 - 35]", "[36 - 45]", "[46 - 54]", "[55 - 65]", "[66+"]
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image /= 255.
  bt_prediction = vgg16.predict(image)
  model = load_model()
  preds = model.predict_proba(bt_prediction)
  return preds


def prepare_data ():
    start = datetime.datetime.now()

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train)

    np.save(btneck_file, bottleneck_features_train)
    end = datetime.datetime.now()
    elapsed = end - start
    print("Czas: ", elapsed)


# trenowanie modelu
def train_model(train_data, num_classes, train_labels, validation_data, validation_labels):
    start = datetime.datetime.now()
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=["acc"])

    history = model.fit(train_data, train_labels,
                        epochs=8,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] dokładność: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] strata: {}".format(eval_loss))
    end = datetime.datetime.now()
    elapsed = end - start
    print("Czas: ", elapsed)

    # tf.saved_model.save(model, 'saved_model/genders-3')
    # model.save(SAVE_PATH)
    print("Zapisano model!")

    return history, model


def main_age():
    # To uruchamiamy tylko za pierwszym razem (zajmuje długo)
    # prepare_data()
    #
    generator_top = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
    shuffle = False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)
    train_data = np.load(btneck_file)
    train_labels = generator_top.classes
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle = False)

    nb_validation_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)
    validation_data = np.load(btneck_file)
    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    history, model = train_model(train_data, num_classes, train_labels, validation_data, validation_labels)

    # Wynik treningu
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))
    plt.plot(epochs, acc, "r", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    print(test_single_image("/home/adam/Desktop/test.jpg", model=model))
