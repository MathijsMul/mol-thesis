#!/bin/bash

python3 main.py './data/binary/neg_det1_noun1_verb_noun2/binary2_4negs_train.txt' './data/binary/neg_det1_noun1_verb_noun2/binary2_4negs_test.txt' tRNN 50 > binary_4negs_trnn.txt &

python3 main.py './data/binary/neg_det1_noun1_verb_noun2/binary2_4negs_train.txt' './data/binary/neg_det1_noun1_verb_noun2/binary2_4negs_test.txt' tRNTN 50 > binary_4negs_trntn.txt &
