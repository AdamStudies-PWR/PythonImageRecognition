from genders import use_model
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# PATH = sys.argv[1]
PATH = "/run/media/adam/DLinux/BackupRIPO/14g1a61.png"

OUTPUT = ["Chłopczyk", "Dziewczynka"]

if not os.path.exists(PATH):
    print("Niepoprawna ścieżka! Spróbuj ponownie")
    exit(0)

image = plt.imread(PATH)
plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.show()
output = use_model(PATH)
print(str(output))
print(str(output[0][0]))
# print(OUTPUT[int(output[0][0])])