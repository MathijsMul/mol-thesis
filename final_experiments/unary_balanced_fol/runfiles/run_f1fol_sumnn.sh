#!/bin/bash

train_file='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/unary_balanced_fol_train.txt'
test_file='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/unary_balanced_fol_test.txt'

runs='1 2 3 4 5'

for run in $runs
do
  echo 'run: '$run

  log_file='sumnn_unary_balanced_fol_'$run'.txt'
  python3 main.py $train_file $test_file sumNN 50 $run > $log_file
done

python3 main.py '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/unary_balanced_fol_train.txt' '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/unary_balanced_fol_test.txt' sumNN 50 test1