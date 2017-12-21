import random

def sample_file(file_in, ratio):
    file_out = file_in + '_downsampled_' + str(ratio)
    with open(file_in, 'r') as fin:
        with open(file_out, 'w') as fout:
            for idx, line in enumerate(fin):
                if random.random() < ratio:
                    fout.write(line)

sample_file('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/diagnose/diagnostic_gru_2dets4negs_pos.txt', 0.4)

