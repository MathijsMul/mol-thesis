import random

def sample_file(file_in, ratio):
    file_out = file_in + '_downsampled_' + str(ratio)
    with open(file_in, 'r') as fin:
        with open(file_out, 'w') as fout:
            for idx, line in enumerate(fin):
                if random.random() < ratio:
                    fout.write(line)

sample_file('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_test_translated_from_nl.txt', 0.01)

