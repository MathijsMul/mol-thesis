import random

in_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
out_file = 'binary_2dets_4negs_test_negations.txt'

downsample_ratio = 1

with open(out_file, 'w') as fout:
    with open(in_file, 'r') as fin:
        for idx, item in enumerate(fin):
            if random.random() < downsample_ratio:
                s = item.split('\t')
                s1 = s[1].strip().split()
                print(s1)

                if s1[2] in ['some', 'all']:
                    # add negation
                    s2 = s1[:2] + ['(', 'not'] + [s1[2]] + [')'] + s1[3:]
                else:
                    # remove negation
                    s2 = s1[:2] + [s1[4]] + s1[6:]
                print(s2)
                s1 = ' '.join(s1)
                s2 = ' '.join(s2)

                fout.write('^' + '\t' + s1 + '\t' + s2 + '\n')


