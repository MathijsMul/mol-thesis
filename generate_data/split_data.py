"""
receive single file with all data, and split this up into training + test,
so that data generation does not have to be performed again for each new train+test set.
can be restricted to specific sentence lengths
"""

import random
import re
from fol_gen_complex import all_sentences

INDY_DOWNSAMPLE_RATIO = 0.008
DOWNSAMPLE_RATIO = 1
TRAIN_LENGTHS = [5, 7, 8]
TEST_LENGTHS = [6, 9]
FILENAME_STEM = 'binary_2dets_4negs_train578_test69'

bulk_file = '/Users/mathijs/Documents/Studie/MoL/thesis/big_files/2det_4neg/bulk_2dets_4negs_combined.txt'

def parse_line(line):
    """
    :param line:
    :return: [relation, premise, hypothesis]
    """
    split = line.split('\t')

    if '' in split:
        split.remove('')

    relation = split[0]
    length1 = split[1].count('(') + 1
    length2 = split[2].count('(') + 1

    def parse_sentence(sentence):
        s = re.sub("\(", "", sentence)
        s = re.sub("\)", "", s).split()

        if s[0] != 'not':
            s.insert(0, '')
        if s[2] != 'not':
            s.insert(2, '')
        if s[4] != 'not':
            s.insert(4, '')
        if s[6] != 'not':
            s.insert(6, '')
        if s[8] != 'not':
            s.insert(8, '')
        return(tuple(s))

    premise, hypothesis = parse_sentence(split[1]), parse_sentence(split[2])
    return([relation, premise, hypothesis, length1, length2])

def split(bulk_f, filename_stem):
    training_file = open(filename_stem + "_train.txt", 'w')
    test_file = open(filename_stem + "_test.txt", 'w')
    sentences = set()

    for counter, s in enumerate(all_sentences()):
        sentences.add(s)

    sentence_list = list(sentences)
    random.shuffle(sentence_list)

    #test_examples = sentence_list[1:int(.33 * len(sentence_list))]  # originally: 0.33

    with open(bulk_f, 'r') as bulk:
        for idx, line in enumerate(bulk):
            [rel, premise, hypothesis, length1, length2] = parse_line(line)

            if random.random() < DOWNSAMPLE_RATIO:
                if (rel != '#'):
                    #if premise in test_examples and hypothesis in test_examples:
                    if ((length1 in TEST_LENGTHS) and (length2 in TEST_LENGTHS)):
                        #if det_subj1 == DET_SUBJ1 and det_subj2 == DET_SUBJ2:
                        test_file.write(line)
                    #elif not (premise in test_examples or hypothesis in test_examples):
                    elif ((length1 in TRAIN_LENGTHS) and (length2 in TRAIN_LENGTHS)):
                        training_file.write(line)

                elif random.random() < INDY_DOWNSAMPLE_RATIO:
                    #if premise in test_examples and hypothesis in test_examples:
                    if ((length1 in TEST_LENGTHS) and (length2 in TEST_LENGTHS)):
                        test_file.write(line)
                    #elif not (premise in test_examples or hypothesis in test_examples):
                    elif ((length1 in TRAIN_LENGTHS) and (length2 in TRAIN_LENGTHS)):
                        training_file.write(line)

        training_file.close()
        test_file.close()

split(bulk_file, FILENAME_STEM)