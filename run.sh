#!/bin/bash

python3 main.py './data/binary/negate_det1/split/binary1_neg_det1_train.txt' './data/binary/negate_det1/split/binary1_neg_det1_test.txt' sumNN 1 > binary_neg_det1_sumnn.txt &

python3 main.py './data/binary/negate_noun1/split/binary1_neg_noun1_train.txt' './data/binary/negate_noun1/split/binary1_neg_noun1_test.txt' sumNN 1 > binary_neg_noun1_sumnn.txt &

python3 main.py './data/binary/negate_verb/split/binary1_neg_verb_train.txt' './data/binary/negate_verb/split/binary1_neg_verb_test.txt' sumNN 1 > binary_neg_verb_sumnn.txt &

python3 main.py './data/binary/neg_det1_noun1_verb_noun2/binary2_4negs_train.txt' './data/binary/neg_det1_noun1_verb_noun2/binary2_4negs_test.txt' sumNN 1 > binary_4negs_sumnn.txt &
