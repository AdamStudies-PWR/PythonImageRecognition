import tensorflow as tf
import os


# MALES
MDIR = "/run/media/adam/DLinux/ripo/Croped/train/male/"
# FEMALES
FDIR = "/run/media/adam/DLinux/ripo/Croped/train/female/"

def process_files(path):
    name = os.path.basename(path)
    temp = name.split("a")
    age = temp[1].split(".")
    return age[0]


class Images:
    def __init__(self, gender, age, path):
        self.gender = gender
        self.age = age
        self.path = path

    def __str__(self):
        return self.gender + str(self.age) + self.path

    def getGender(self): return self.gender
    def getAge(self): return self.age
    def getPath(self): return self.path


image_list = []

for file in os.listdir(MDIR):
    temp = Images("male", process_files(file), MDIR + file)
    image_list.append(temp)

for file in os.listdir(FDIR):
    temp = Images("female", process_files(file), FDIR + file)
    image_list.append(temp)


for check in image_list:
    print(check)
