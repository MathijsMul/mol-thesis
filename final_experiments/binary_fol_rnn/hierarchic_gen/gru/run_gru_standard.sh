#!/bin/bash

main_dir='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/hierarchic_gen/no_brackets/from_standard/'

det_subj_pairs='somesome someall somenot_some somenot_all allsome allall allnot_some allnot_all not_somesome not_someall not_somenot_some not_somenot_all not_allsome not_allall not_allnot_some not_allnot_all'
runs='1 2 3 4 5'

for det_subj_pair in $det_subj_pairs
do
  echo $det_subj_pair

  for run in $runs
  do
    echo $run
      train_file=$main_dir'train/binary_2dets_4negs_train_'$det_subj_pair'_0bracket_pairs.txt'
      test_file=$main_dir'test/binary_2dets_4negs_test_'$det_subj_pair'_0bracket_pairs.txt'

      log_file='gru_standard'$det_subj_pair'_'$run'.txt'
      echo $train_file
      echo $test_file
      echo $log_file
      python3 main.py $train_file $test_file GRU 50 $run > $log_file
  done
done
