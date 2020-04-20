import os
import sys

if len(sys.argv) != 2:
    print("Niepoprawana ilość argumentów")
    exit(0)

# TU WSTAWIC SCIEZKE DO OBRAZÓW TRENINGOWYCH
DB = sys.argv[1]
# TU WSTAWIC SCIEZKE DO OBRAZÓW TRENINGOWYCH
if os.path.exists(DB):
    images = len(os.listdir(DB))
    if images == 0:
        print("Brak obrazów! Sprawdź czy podano poprawną ścieżkę!")
        exit(0)
    else:
        print("Zanleziono ", images, " obrazów do treningu")
else:
    print("Niepoprawna ścieżka! Spróbuj ponownie")
    exit(0)

males = DB + "male"
females = DB + "female"

if not os.path.exists(males): os.mkdir(males)
if not os.path.exists(females): os.mkdir(females)

try:
    for filename in os.listdir(DB):
        if filename != "male" and filename != "female":
            if not filename.find('g1') == -1:
                os.replace(DB + filename, males + "/" + filename)
            else:
                os.replace(DB + filename, females + "/" + filename)
except:
    print("Ding Dong something wrong!")
    print(DB)
    print(females)
    print(males)