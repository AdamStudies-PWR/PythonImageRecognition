import os
import sys
import shutil


def processFiles(name):
    temp = name.split("a")
    age = temp[1].split(".")
    return age[0]


PATH = sys.argv[1]

if not os.path.exists(PATH):
    print("Niepoprawna ścieżka! Spróbuj ponownie")
    exit(0)

#INPUT
# W tej linijce ustawić ścięzkę do treningu przygotowanych wcześniej do rozpoznawania wieku
ITRAIN = "/media/adam/DLinux/ripo/Croped/train/"
# W tej linijce ustawić ścięzkę do walidacji przygotowanych wcześniej do rozpoznawania wieku
IVALIDATE = "/media/adam/DLinux/ripo/Croped/validation/"

TRAINM = ITRAIN + "/female"
TRAINF = ITRAIN + "/male"
VALIDM = IVALIDATE + "/female"
VALIDF = IVALIDATE + "/male"

OUTPUT = "/media/adam/DLinux/ripo/Age/"

cat1 = PATH + "/Age/0-8"
cat2 = PATH + "/Age/9-14"
cat3 = PATH + "/Age/15-23"
cat4 = PATH + "/Age/24-35"
cat5 = PATH + "/Age/36-45"
cat6 = PATH + "/Age/46-54"
cat7 = PATH + "/Age/55-65"
cat8 = PATH + "/Age/66+"

if not os.path.exists(cat1): os.mkdir(PATH + "/Age")
if not os.path.exists(cat1): os.mkdir(cat1)
if not os.path.exists(cat2): os.mkdir(cat2)
if not os.path.exists(cat3): os.mkdir(cat3)
if not os.path.exists(cat4): os.mkdir(cat4)
if not os.path.exists(cat5): os.mkdir(cat5)
if not os.path.exists(cat6): os.mkdir(cat6)
if not os.path.exists(cat7): os.mkdir(cat7)
if not os.path.exists(cat8): os.mkdir(cat8)

age = 0

for filename in os.listdir(TRAINM):
    age = processFiles(filename)
    if int(age) < 9:
        shutil.copy(TRAINM + "/" + filename, cat1)
    elif int(age) < 15:
        shutil.copy(TRAINM + "/" + filename, cat2)
    elif int(age) < 24:
        shutil.copy(TRAINM + "/" + filename, cat3)
    elif int(age) < 36:
        shutil.copy(TRAINM + "/" + filename, cat4)
    elif int(age) < 46:
        shutil.copy(TRAINM + "/" + filename, cat5)
    elif int(age) < 55:
        shutil.copy(TRAINM + "/" + filename, cat6)
    elif int(age) < 66:
        shutil.copy(TRAINM + "/" + filename, cat7)
    else:
        shutil.copy(TRAINM + "/" + filename, cat8)

for filename in os.listdir(TRAINF):
    age = processFiles(filename)
    if int(age) < 9:
        shutil.copy(TRAINF + "/" + filename, cat1)
    elif int(age) < 15:
        shutil.copy(TRAINF + "/" + filename, cat2)
    elif int(age) < 24:
        shutil.copy(TRAINF + "/" + filename, cat3)
    elif int(age) < 36:
        shutil.copy(TRAINF + "/" + filename, cat4)
    elif int(age) < 46:
        shutil.copy(TRAINF + "/" + filename, cat5)
    elif int(age) < 55:
        shutil.copy(TRAINF + "/" + filename, cat6)
    elif int(age) < 66:
        shutil.copy(TRAINF + "/" + filename, cat7)
    else:
        shutil.copy(TRAINF + "/" + filename, cat8)

for filename in os.listdir(VALIDM):
    age = processFiles(filename)
    if int(age) < 9:
        shutil.copy(VALIDM + "/" + filename, cat1)
    elif int(age) < 15:
        shutil.copy(VALIDM + "/" + filename, cat2)
    elif int(age) < 24:
        shutil.copy(VALIDM + "/" + filename, cat3)
    elif int(age) < 36:
        shutil.copy(VALIDM + "/" + filename, cat4)
    elif int(age) < 46:
        shutil.copy(VALIDM + "/" + filename, cat5)
    elif int(age) < 55:
        shutil.copy(VALIDM + "/" + filename, cat6)
    elif int(age) < 66:
        shutil.copy(VALIDM + "/" + filename, cat7)
    else:
        shutil.copy(VALIDM + "/" + filename, cat8)

for filename in os.listdir(VALIDF):
    age = processFiles(filename)
    if int(age) < 9:
        shutil.copy(VALIDF + "/" + filename, cat1)
    elif int(age) < 15:
        shutil.copy(VALIDF + "/" + filename, cat2)
    elif int(age) < 24:
        shutil.copy(VALIDF + "/" + filename, cat3)
    elif int(age) < 36:
        shutil.copy(VALIDF + "/" + filename, cat4)
    elif int(age) < 46:
        shutil.copy(VALIDF + "/" + filename, cat5)
    elif int(age) < 55:
        shutil.copy(VALIDF + "/" + filename, cat6)
    elif int(age) < 66:
        shutil.copy(VALIDF + "/" + filename, cat7)
    else:
        shutil.copy(VALIDF + "/" + filename, cat8)

print("Obrazy trzeba ręcznie podzielić na trening i walidacje!")

