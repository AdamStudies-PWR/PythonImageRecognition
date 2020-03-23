from genders import use_model
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = sys.argv[1]

if not os.path.exists(PATH):
    print("Niepoprawna ścieżka! Spróbuj ponownie")
    exit(0)

image = plt.imread(PATH)
plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.show()
use_model(PATH)
