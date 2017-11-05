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

echo TRNN, BINARY DATA
python3 main.py './data/binary/negate_verb/split/binary1_neg_verbtrain.txt' './data/binary/negate_verb/split/binary1_neg_verbtest.txt' False > binary_neg_verb_trnn.txt

echo TRNTN, BINARY DATA
python3 main.py './data/binary/negate_verb/split/binary1_neg_verbtrain.txt' './data/binary/negate_verb/split/binary1_neg_verbtest.txt' True > binary_neg_verb_trntn.txt


