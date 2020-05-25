import os
import shutil

def processFiles(name):
    temp = name.split("a")
    age = temp[1].split(".")
    return age[0]

#INPUT
ITRAIN = "/media/adam/DLinux/ripo/Age/train/"
IVALIDATE = "/media/adam/DLinux/ripo/Age/validation/"

OUTPUT = "/media/adam/DLinux/ripo/Age/"

cat1 = "/media/adam/DLinux/ripo/Age/0-8"
cat2 = "/media/adam/DLinux/ripo/Age/9-14"
cat3 = "/media/adam/DLinux/ripo/Age/15-23"
cat4 = "/media/adam/DLinux/ripo/Age/24-35"
cat5 = "/media/adam/DLinux/ripo/Age/36-45"
cat6 = "/media/adam/DLinux/ripo/Age/46-54"
cat7 = "/media/adam/DLinux/ripo/Age/55-65"
cat8 = "/media/adam/DLinux/ripo/Age/66+"

if not os.path.exists(cat1): os.mkdir(cat1)
if not os.path.exists(cat2): os.mkdir(cat2)
if not os.path.exists(cat3): os.mkdir(cat3)
if not os.path.exists(cat4): os.mkdir(cat4)
if not os.path.exists(cat5): os.mkdir(cat5)
if not os.path.exists(cat6): os.mkdir(cat6)
if not os.path.exists(cat7): os.mkdir(cat7)
if not os.path.exists(cat8): os.mkdir(cat8)

for filename in os.listdir(ITRAIN):
    if int(filename) < 9:
        for file in os.listdir(ITRAIN + filename):
            shutil.copy(ITRAIN + filename + "/" + file, cat1)
    elif int(filename) < 15:
        for file in os.listdir(ITRAIN + filename):
            shutil.copy(ITRAIN + filename + "/" + file, cat2)
    elif int(filename) < 24:
        for file in os.listdir(ITRAIN + filename):
            shutil.copy(ITRAIN + filename + "/" + file, cat3)
    elif int(filename) < 36:
        for file in os.listdir(ITRAIN + filename):
            shutil.copy(ITRAIN + filename + "/" + file, cat4)
    elif int(filename) < 46:
        for file in os.listdir(ITRAIN + filename):
            shutil.copy(ITRAIN + filename + "/" + file, cat5)
    elif int(filename) < 55:
        for file in os.listdir(ITRAIN + filename):
            shutil.copy(ITRAIN + filename + "/" + file, cat6)
    elif int(filename) < 66:
        for file in os.listdir(ITRAIN + filename):
            shutil.copy(ITRAIN + filename + "/" + file, cat7)
    else:
        for file in os.listdir(ITRAIN + filename):
            shutil.copy(ITRAIN + filename + "/" + file, cat8)
            
for filename in os.listdir(IVALIDATE):
    if int(filename) < 9:
        for file in os.listdir(IVALIDATE + filename):
            shutil.copy(IVALIDATE + filename + "/" + file, cat1)
    elif int(filename) < 15:
        for file in os.listdir(IVALIDATE + filename):
            shutil.copy(IVALIDATE + filename + "/" + file, cat2)
    elif int(filename) < 24:
        for file in os.listdir(IVALIDATE + filename):
            shutil.copy(IVALIDATE + filename + "/" + file, cat3)
    elif int(filename) < 36:
        for file in os.listdir(IVALIDATE + filename):
            shutil.copy(IVALIDATE + filename + "/" + file, cat4)
    elif int(filename) < 46:
        for file in os.listdir(IVALIDATE + filename):
            shutil.copy(IVALIDATE + filename + "/" + file, cat5)
    elif int(filename) < 55:
        for file in os.listdir(IVALIDATE + filename):
            shutil.copy(IVALIDATE + filename + "/" + file, cat6)
    elif int(filename) < 66:
        for file in os.listdir(IVALIDATE + filename):
            shutil.copy(IVALIDATE + filename + "/" + file, cat7)
    else:
        for file in os.listdir(IVALIDATE + filename):
            shutil.copy(IVALIDATE + filename + "/" + file, cat8)
