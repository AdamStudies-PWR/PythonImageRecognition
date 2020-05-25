# extract and plot each detected face in a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN


def crop_faces(filename):
    # load image from file
    data = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(data)
    cropped_faces = []
    for i in range(len(faces)):
        # get coordinates
        x1, y1, width, height = faces[i]['box']
        x2, y2 = x1 + width, y1 + height
        cropped_faces.append(data[y1:y2, x1:x2])
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
# # detect faces in the image
# faces = crop_faces(filename)
# # display faces on the original image
# draw_faces(faces)
