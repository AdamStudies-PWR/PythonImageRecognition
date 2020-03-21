from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

IMG_HEIGHT = 200
IMG_WIDTH = 200

if len(sys.argv) != 2:
    print("Nipoprawana ilość argumentów")
    exit(0)

# TU WSTAWIC SCIEZKE DO OBRAZÓW TRENINGOWYCH
DB = sys.argv[1]
# TU WSTAWIC SCIEZKE DO OBRAZÓW TRENINGOWYCH
if os.path.isdir(DB):
    images = len(os.listdir(DB))
    if images == 0:
        print("Brak obrazów! Sprawdź czy podano poprawną ścieżkę!")
        exit(0)
    else:
        print("Zanleziono ", images, " obrazów do treningu")
else:
    print("Niepoprawna ścieżka! Spróbuj ponownie")
    exit(0)
