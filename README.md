# PythonImageRecognition
Projekt na **Rozpowanie i przetwarzanie obrazów** [RIPO] na Politechnikę Wrocławską.

Prowadzący: Dr inż. Jacek Mazurkiewicz.

## Założenia projektu:

Rozpoznawanie wieku i płci na podstawie zdjęcia lub obrazu an żywo z kamery.

## Wykorzystane dane: 

Wykorzystaliśmy bazę danych IMDb zawierająco opisane zdjęcia aktórów (wiek + płeć) dostępną [tutaj](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

## Rezultaty

Udało sie zrealizować rozpoznawanie płci, niestety jednak rozpoznawanie wieku nie działa poprawnie - model [wieku](https://github.com/AdamStudies-PWR/PythonImageRecognition/tree/age_recognition) tworzy się poprawnie ale nie udało nam się go zapisać do późniejszego wyokrzystania.

## Rozpoznawianie wieku

Stworzony przez nas model wieku poprawnie rozpaznywał wiek na testowanych przez na obrazkach. Jednakże napotkaliśmy problemy z zapisem stworznego przez nas modelu. W związku z tym nie udało nam się zaimplementować tego modułu w głównej aplikacji. 

Główna część projektu dostępna jest [tutaj](https://github.com/AdamStudies-PWR/PythonImageRecognition/tree/master)
