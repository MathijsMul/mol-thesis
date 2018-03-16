python3 main.py '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/bowman/f1_train.txt' '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/bowman/f1_test.txt' sumNN 50 bowman_rep_all_epochs > 'sumnn_f1_all_epochs.txt' &

python3 main.py '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/bowman/f1_train.txt' '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/bowman/f1_test.txt' tRNN 50 bowman_rep_all_epochs > 'trnn_f1_all_epochs.txt' &

python3 main.py '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/bowman/f1_train.txt' '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/bowman/f1_test.txt' tRNTN 50 bowman_rep_all_epochs > 'trntn_f1_all_epochs.txt' &
