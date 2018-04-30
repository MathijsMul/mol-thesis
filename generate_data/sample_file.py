"""
sample data
"""

import random

def sample_file(file_in, ratio):
    file_out = file_in + str(ratio)
    with open(file_in, 'r') as fin:
        with open(file_out, 'w') as fout:
            for idx, line in enumerate(fin):
                if random.random() < ratio:
                    fout.write(line)

f = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/generate_data/binary_2dets_4negs_train578_test69_train.txt'
ratio = 0.33
sample_file(f, ratio)

g = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/generate_data/binary_2dets_4negs_train578_test69_test.txt'
ratio = 0.25
sample_file(g, ratio)
