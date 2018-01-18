#!/bin/bash

#main_dir='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/hierarchic_gen/'
main_dir='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/hierarchic_gen/segment_bulk_2det_4negs/'

det_subj_pairs='somesome someall somenot_some somenot_all allsome allall allnot_some allnot_all not_somesome not_someall not_somenot_some not_somenot_all not_allsome not_allall not_allnot_some not_allnot_all'

for det_subj_pair in $det_subj_pairs
do
  echo $det_subj_pair

  #train_file=$main_dir'train/binary_2dets_4negs_train_'$det_subj_pair'.txt'
  train_file=$main_dir'train/binary_2dets_4negs_bulk_'$det_subj_pair'_train.txt'

  #test_file=$main_dir'test/binary_2dets_4negs_test_'$det_subj_pair'.txt'
  test_file=$main_dir'test/binary_2dets_4negs_bulk_'$det_subj_pair'_test.txt'

  log_file='binary_2dets_4negs_gru_frombulk_'$det_subj_pair'.txt'
  echo $train_file
  echo $test_file
  echo $log_file
  python3 main.py $train_file $test_file GRU 50 1 > $log_file
done

python3 main.py '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_train_downs0.2.txt' '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test_downs0.2.txt' GRU 50 frombulk1 > binary_2dets_4negs_gru_frombulk1.txt