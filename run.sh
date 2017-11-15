#!/bin/bash

python3 main.py './data/binary/2dets_4negs/binary_2dets_4negs_train.txt' './data/binary/2dets_4negs/binary_2dets_4negs_test.txt' sumNN 1 > bin_2dets_4negs_sumnn.txt &

python3 main.py './data/binary/2dets_4negs/binary_2dets_4negs_train.txt' './data/binary/2dets_4negs/binary_2dets_4negs_test.txt' tRNN 1 > bin_2dets_4negs_trnn.txt &

python3 main.py './data/binary/2dets_4negs/binary_2dets_4negs_train.txt' './data/binary/2dets_4negs/binary_2dets_4negs_test.txt' tRNTN 1 > bin_2dets_4negs_trntn.txt &