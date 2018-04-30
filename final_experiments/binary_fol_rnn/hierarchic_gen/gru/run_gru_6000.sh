#!/bin/bash

train_file='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_train_6000.txt'
test_file='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test_downs1500.txt'
runs='1 2 3 4 5'

for run in $runs
do
  echo $run

  log_file='gru_6000_'$run'.txt'
  echo $train_file
  echo $test_file
  echo $log_file
  python3 main.py $train_file $test_file GRU 50 $run > $log_file
done
