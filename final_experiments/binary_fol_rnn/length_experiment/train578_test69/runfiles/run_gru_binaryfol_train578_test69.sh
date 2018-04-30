#!/bin/bash

train_file='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/train578_test69/binary_2dets_4negs_train578_test69_train_0bracket_pairs.txt'
test_file='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/train578_test69/binary_2dets_4negs_train578_test69_test_0bracket_pairs.txt'

runs='1 2 3 4 5'

for run in $runs
do
  echo 'run: '$run

  log_file='gru_binaryfol_train578_test69_'$run'.txt'
  python3 main.py $train_file $test_file GRU 50 $run > $log_file
done