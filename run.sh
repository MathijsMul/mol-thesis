#!/bin/bash

python3 main.py './data/binary/2dets_4negs/578/binary_2dets_4negs_578_train.txt' './data/binary/2dets_4negs/578/binary_2dets_4negs_56789_test.txt' sumNN 50 > bin_2dets_4negs_578_sumnn.txt &

python3 main.py './data/binary/2dets_4negs/578/binary_2dets_4negs_578_train.txt' './data/binary/2dets_4negs/578/binary_2dets_4negs_56789_test.txt' tRNN 50 > bin_2dets_4negs_578_trnn.txt &

python3 main.py './data/binary/2dets_4negs/578/binary_2dets_4negs_578_train.txt' './data/binary/2dets_4negs/578/binary_2dets_4negs_56789_test.txt' tRNTN 50 > bin_2dets_4negs_578_trntn.txt &
