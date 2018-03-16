"""
reduce relative frequency of #
"""

import random

def get_rel(line):
    """
    :param line:
    :return: relation
    """
    split = line.split('\t')

    if '' in split:
        split.remove('')

    relation = split[0]
    return(relation)

def reduce_indy(filename, ratio):
    file_out = filename + '_inddown_' + str(ratio)
    fout = open(file_out, 'w')

    with open(filename, 'r') as bulk:
        for idx, line in enumerate(bulk):
            rel = get_rel(line)
            if rel != '#':
                fout.write(line)
            else:
                if random.random() < ratio:
                    fout.write(line)

        fout.close()

f='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_data1_animals_test.txt'
reduce_indy(f, 0.7)

