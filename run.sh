#!/bin/bash

#python3 main.py './data/binary/2dets_4negs/binary_2dets_4negs_train.txt' './data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' SRN 50 > binary_2dets_4negs_srn.txt &

python3 main.py './data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0.5bracket_pairs.txt' './data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' SRN 50 > binary_2dets_4negs_0.5brackets_srn.txt &

python3 main.py './data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0bracket_pairs.txt' './data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' SRN 50 > binary_2dets_4negs_0brackets_srn.txt &

python3 main.py './data/binary/2dets_4negs/binary_2dets_4negs_train.txt' './data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' GRU 50 > binary_2dets_4negs_gru.txt &

python3 main.py './data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0.5bracket_pairs.txt' './data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' GRU 50 > binary_2dets_4negs_0.5brackets_gru.txt &

python3 main.py './data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0bracket_pairs.txt' './data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' GRU 50 > binary_2dets_4negs_0brackets_gru.txt &