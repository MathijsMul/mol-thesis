"""
receive single file with all data, and split this up into training + test,
so that data generation does not have to be performed again for each new train+test set
"""

import random
import re
from fol_gen_complex import all_sentences

INDY_DOWNSAMPLE_RATIO = 0.025
FILENAME_STEM = 'binary1_'

bulk_file = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/binary_all.txt'

training_file = open(FILENAME_STEM + "train.txt", 'w')
test_file = open(FILENAME_STEM + "test.txt", 'w')

sentences = set()

for counter, s in enumerate(all_sentences()):
    sentences.add(s)

sentence_list = list(sentences)
random.shuffle(sentence_list)
test_examples = sentence_list[1:int(.33 * len(sentence_list))]

def parse_line(line):
    """

    :param line:
    :return: [relation, premise, hypothesis]
    """
    # TODO: translate line in file to sentence in format of sentences in test_examples

    split = line.split('\t')
    #print(split)
    relation = split[0]

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

    #[not] quantifier [not] subject [not] verb [not] quantifier [not] object

    premise, hypothesis = parse_sentence(split[1]), parse_sentence(split[2])

    return([relation, premise, hypothesis])

with open(bulk_file, 'r') as bulk:
    for idx, line in enumerate(bulk):

        [rel, premise, hypothesis] = parse_line(line)
        #print([rel, premise, hypothesis] )

        if (rel != '#'):
            if premise in test_examples and hypothesis in test_examples:
                test_file.write(line)
            elif not (premise in test_examples or hypothesis in test_examples):
                training_file.write(line)

        elif random.random() < INDY_DOWNSAMPLE_RATIO:
            if premise in test_examples and hypothesis in test_examples:
                test_file.write(line)
            elif not (premise in test_examples or hypothesis in test_examples):
                training_file.write(line)

    training_file.close()
    test_file.close()

#bulk_file.close()