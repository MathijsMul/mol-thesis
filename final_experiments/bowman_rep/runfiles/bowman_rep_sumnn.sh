#!/bin/bash

data_dir='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/bowman/'

folds='f1 f2 f3 f4 f5'

for fold in $folds
do
  echo 'fold: '$fold

  train_file=$data_dir$fold'_train.txt'
  test_file=$data_dir$fold'_test.txt'

  runs='1 2 3 4 5'
  for run in $runs
  do
    echo 'run: '$run
    log_file=$fold'_'$run'sumnn_bowman_rep.txt'
    python3 main.py $train_file $test_file sumNN 50 $run > $log_file
  done

done