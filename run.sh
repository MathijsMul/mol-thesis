#!/bin/bash

python3 main.py 'data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0.5bracket_pairs.txt' 'data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' SRN 50 1nodrop > binary_2dets_4negs_0.5brack_nodrop_srn1.txt &

python3 main.py 'data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0.5bracket_pairs.txt' 'data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' GRU 50 1nodrop > binary_2dets_4negs_0.5brack_nodrop_gru1.txt &

python3 main.py 'data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0.5bracket_pairs.txt' 'data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' LSTM 50 1nodrop > binary_2dets_4negs_0.5brack_nodrop_lstm1.txt &


python3 main.py 'data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0bracket_pairs.txt' 'data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' SRN 50 1nodrop > binary_2dets_4negs_0brack_nodrop_srn1.txt &

python3 main.py 'data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0bracket_pairs.txt' 'data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' GRU 50 1nodrop > binary_2dets_4negs_0brack_nodrop_gru1.txt &

python3 main.py 'data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0bracket_pairs.txt' 'data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt' LSTM 50 1nodrop > binary_2dets_4negs_0brack_nodrop_lstm1.txt &


python3 main.py 'data/unary/nl/nl_data1_animals_train.txt' 'data/unary/nl/nl_data1_animals_test.txt' GRU 50 1 > unary_nl_animals_gru1.txt

python3 main.py 'data/binary/2dets_4negs/binary_2dets_4negs_train.txt' 'data/binary/2dets_4negs/binary_2dets_4negs_test.txt' GRU_connected 50 1 > binary_2dets_4negs_gru_conn1.txt

python3 main.py 'data/binary/2dets_4negs/binary_2dets_4negs_train.txt' 'data/binary/2dets_4negs/binary_2dets_4negs_test.txt' GRU 50 2layers > binary_2dets_4negs_gru_2layers.txt