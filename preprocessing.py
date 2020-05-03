import os

from crop_face import crop_faces

temp_path = "/media/adam/DLinux/ripo/Croped/train/male"
for filename in os.listdir(temp_path):
    test = crop_faces(temp_path + "/" + filename)
    print(len(test))