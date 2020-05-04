# extract and plot each detected face in a photograph
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np

# PATH = sys.argv[1]


# if not os.path.exists(PATH):
#     print("Niepoprawna ścieżka! Spróbuj ponownie")
#     exit(0)


def crop_faces(filename):
    # load image from file
    data = np.asarray(Image.open(filename))  # pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    cropped_faces = []
    try:
        faces = detector.detect_faces(data)
        for i in range(len(faces)):
            # get coordinates
            x1, y1, width, height = faces[i]['box']
            x2, y2 = x1 + width, y1 + height
            cropped_faces.append(data[y1:y2, x1:x2])
    except:
        print("Błąd! Plik: ", filename)

    return cropped_faces

# draw each face separately


def draw_faces(cropped_faces):
    for i in range(len(cropped_faces)):
        # define subplot
        pyplot.subplot(1, len(cropped_faces), i+1)
        pyplot.axis('off')
        # plot face
        pyplot.imshow(cropped_faces[i])
    pyplot.show()


# filename = 'test1.jpg'
# detect faces in the image
# faces = crop_faces(PATH)
# display faces on the original image
# draw_faces(faces)
