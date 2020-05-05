from genders import fit_model_from_path, read_history
import sys

PATH = sys.argv[1]

BATCH_SIZE = 32
EPOCHS = 0
IMG_HEIGHT = 200
IMG_WIDTH = 200
CHECKPOINT_PATH = 'training_1/cp.ckpt'
SAVE_PATH = 'saved_model/genders'
# Muszę tak podawać ścieżke bo inaczej zapisuje mi w home xD
# CHECKPOINT_PATH = '/home/adam/RIPO/PythonImageRecognition/training_1/cp.ckpt'
# SAVE_PATH = '/home/adam/RIPO/PythonImageRecognition/saved_model/genders'

history = fit_model_from_path(path_data=PATH, epochs=EPOCHS,
                              batch_size=BATCH_SIZE, save_path=SAVE_PATH, checkpoint_path=CHECKPOINT_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH))

read_history(history, EPOCHS)
