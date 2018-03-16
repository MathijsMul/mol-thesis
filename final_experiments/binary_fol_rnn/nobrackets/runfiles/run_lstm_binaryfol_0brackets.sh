#!/bin/bash

train_file='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/train/binary_2dets_4negs_train_0bracket_pairs.txt'
test_file='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt'

runs='1 2 3 4 5'

for run in $runs
do
  echo 'run: '$run

  log_file='lstm_binaryfol_0brackets_'$run'.txt'
  python3 main.py $train_file $test_file LSTM 50 $run > $log_file
done