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
python3 main.py './data/binary/negate_noun1/split/binary1_neg_noun1train.txt' './data/binary/negate_noun1/split/binary1_neg_noun1test.txt' False > binary_neg_noun1_trnn.txt


echo TRNTN, BINARY DATA
python3 main.py './data/binary/negate_noun1/split/binary1_neg_noun1train.txt' './data/binary/negate_noun1/split/binary1_neg_noun1test.txt' True > binary_neg_noun1_trntn.txt


