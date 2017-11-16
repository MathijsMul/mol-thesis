import random

def sample_file(file_in, ratio):
    file_out = file_in + str(ratio)
    with open(file_in, 'r') as fin:
        with open(file_out, 'w') as fout:
            for idx, line in enumerate(fin):
                if random.random() < ratio:
                    fout.write(line)

file_in = 'binary_2dets_4negs_2457_train.txt'
ratio = 0.75
sample_file(file_in, ratio)

file_in = 'binary_2dets_4negs_2457_test.txt'
ratio = 0.25
sample_file(file_in, ratio)