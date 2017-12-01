#!/bin/bash

#python3 main.py './data/binary/2dets_4negs/train567_test89/binary_2dets_4negs_train567_test89_train.txt' './data/binary/2dets_4negs/train567_test89/binary_2dets_4negs_train567_test89_test.txt' sumNN 50 > bin_2dets_4negs_train567_test89_sumnn.txt &

#python3 main.py './data/binary/2dets_4negs/train567_test89/binary_2dets_4negs_train567_test89_train.txt' './data/binary/2dets_4negs/train567_test89/binary_2dets_4negs_train567_test89_test.txt' tRNN 50 > bin_2dets_4negs_train567_test89_trnn.txt &

#python3 main.py './data/binary/2dets_4negs/train567_test89/binary_2dets_4negs_train567_test89_train.txt' './data/binary/2dets_4negs/train567_test89/binary_2dets_4negs_train567_test89_test.txt' tRNTN 50 > bin_2dets_4negs_train567_test89_trntn.txt &

python3 main.py './data/binary/2dets_4negs/binary_2dets_4negs_train.txt' './data/binary/2dets_4negs/binary_2dets_4negs_test.txt' SRN 50 > binary_2dets_4negs_srn.txt