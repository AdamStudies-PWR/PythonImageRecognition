from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

PATH = sys.argv[1]

BATCH_SIZE = 128
EPOCHS = 3
IMG_HEIGHT = 150
IMG_WIDTH = 150
CHECKPOINT_PATH = 'training_1/cp.ckpt'
SAVE_PATH = 'saved_model/genders'


def load_data_from_path(path=sys.argv[1], batch_size=128, target_size=(150, 150)):

    if not os.path.exists(path):
        print("Niepoprawna ścieżka! Spróbuj ponownie")
        exit(0)

    train_dir = os.path.join(path, "train")
    validation_dir = os.path.join(path, "validation")

    # directory with our training cat pictures
    train_male_dir = os.path.join(train_dir, 'male')
    # directory with our training dog pictures
    train_female_dir = os.path.join(train_dir, 'female')
    # directory with our validation cat pictures
    validation_male_dir = os.path.join(validation_dir, 'male')
    # directory with our validation dog pictures
    validation_female_dir = os.path.join(validation_dir, 'female')

    num_male_tr = len(os.listdir(train_male_dir))
    num_female_tr = len(os.listdir(train_female_dir))

    num_male_val = len(os.listdir(validation_male_dir))
    num_female_val = len(os.listdir(validation_female_dir))

    total_train = num_male_tr + num_female_tr
    total_val = num_male_val + num_female_val

    print('Obrazki treningowe mężczyźni:', num_male_tr)
    print('Obrazki treningowe kobiety:', num_female_tr)

    print('Obrazki sprawdzające mężczyźni:', num_male_val)
    print('Obrazki sprawdzające kobiety:', num_female_val)
    print("--")
    print("W sumie treningowych:", total_train)
    print("W sumie testowych:", total_val)

    train_image_generator = ImageDataGenerator(rescale=1./255)
    validation_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=target_size,
                                                               class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=target_size,
                                                                  class_mode='binary')
    return train_data_gen, val_data_gen, total_train, total_val


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def create_model(target_size=(150, 150)):
    h, w = target_size
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(h, w, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()
    return model


def load_model_from_checkpoint(model, checkpoint_path, train_data_gen):
    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))

    if latest:
        model.load_weights(latest)

        print("Restoring model")
        loss, acc = model.evaluate(train_data_gen, verbose=1)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))


def fit_model_from_path(path_data=PATH, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, save_path=SAVE_PATH, checkpoint_path=CHECKPOINT_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH)):

    train_data_gen, val_data_gen, total_train, total_val = load_data_from_path(
        path_data, batch_size, target_size)

    # sample_training_images, _ = next(train_data_gen)
    # plotImages(sample_training_images[:5])

    model = create_model(target_size)

    load_model_from_checkpoint(model, checkpoint_path, train_data_gen)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)

    history = model.fit(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size,
        callbacks=[cp_callback])  # Pass callback to training

    # tf.saved_model.save(model, 'saved_model/genders-3')
    model.save(save_path)
    print("Saved model")
    return history


def read_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def load_model(path_model):
    return tf.keras.models.load_model(path_model)


def use_model(path_image, path_model=SAVE_PATH):

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    test_image = image.load_img(
        path_image, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    model = load_model(path_model)
    # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    # prediction = probability_model.predict(test_image)
    prediction = model.predict(test_image)
    return prediction
