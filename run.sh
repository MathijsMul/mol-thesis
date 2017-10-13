#!/bin/bash

echo NL ANIMALS
python3 main.py 'data/final/fol/fol_animals_train_translated_from_nl.txt' 'data/final/fol/fol_animals_test_translated_from_nl.txt' > nl_animals_date.txt

echo FOL ANIMALS
python3 main.py 'data/final/fol/fol_animals_train_translated_from_nl.txt' 'data/final/fol/fol_animals_test_translated_from_nl.txt' > fol_animals_date.txt

echo FOL PEOPLE
python3 main.py 'data/final/fol/fol_data1_peopletrain.txt' 'data/final/fol/fol_data1_peopletest.txt' > fol_people_date.txt