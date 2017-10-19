#!/bin/bash

# echo NL ANIMALS
# python3 main.py 'data/final/fol/fol_animals_train_translated_from_nl.txt' 'data/final/fol/fol_animals_test_translated_from_nl.txt' > nl_animals_date.txt

# echo FOL ANIMALS
# python3 main.py 'data/final/fol/fol_animals_train_translated_from_nl.txt' 'data/final/fol/fol_animals_test_translated_from_nl.txt' > fol_animals_date.txt

# echo FOL PEOPLE
# python3 main.py 'data/final/fol/fol_data1_peopletrain.txt' 'data/final/fol/fol_data1_peopletest.txt' > fol_people_date.txt

# binary data

#train_data_file = './data/binary/split/binary1_train.txt'
#test_data_file = './data/binary/split/binary1_test.txt'

echo First try on dummy files
echo TRNN, BINARY DATA
python3 main.py './data/binary_exp2.txt' './data/binary_exp2.txt' False > binary_trnn1.txt

echo TRNTN, BINARY DATA
python3 main.py './data/binary_exp2.txt' './data/binary_exp2.txt' True > binary_trntn1.txt


