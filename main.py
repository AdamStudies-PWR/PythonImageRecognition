from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Seqential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# TU WSTAWIC SCIEZKE DO OBRAZÓW TRENINGOWYCH
DB = "/run/media/adam/DLinux/ripo/Croped/"
# TU WSTAWIC SCIEZKE DO OBRAZÓW TRENINGOWYCH

print("Zanleziono ", DB, " obrazów do treningu")
