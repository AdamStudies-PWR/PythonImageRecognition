from genders import fit_model_from_path, read_history
import sys

PATH = sys.argv[1]

BATCH_SIZE = 128
EPOCHS = 5
IMG_HEIGHT = 300
IMG_WIDTH = 300
CHECKPOINT_PATH = 'training_1/cp.ckpt'
SAVE_PATH = 'saved_model/genders'

history = fit_model_from_path(path_data=PATH, epochs=EPOCHS,
                              batch_size=BATCH_SIZE, save_path=SAVE_PATH, checkpoint_path=CHECKPOINT_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH))

read_history(history)
