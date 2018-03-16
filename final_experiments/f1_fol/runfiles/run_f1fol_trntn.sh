#!/bin/bash

data_dir='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_'

runs='1 2 3 4 5'

for run in $runs
do
  echo 'run: '$run

  train_file=$data_dir'train_translated_from_nl.txt'
  test_file=$data_dir'test_translated_from_nl.txt'

  log_file='folf1_'$run'trntn.txt'
  python3 main.py $train_file $test_file tRNTN 50 $run > $log_file
done