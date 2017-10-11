import quantgen as nl
import gendata_allterms_morequant as fol

from itertools import product
from collections import Counter
import random

import matplotlib.pyplot as plt
import numpy as np
#import plotly.plotly as py

FILE_OUTPUT = False
FILENAME_STEM = "comparison_"
SAMPLES = 10
SAMPLE_SIZE = 1000
VISUALIZE = True

nouns = ['warthogs', 'turtles', 'mammals', 'reptiles', 'pets']
#nouns = ['warthogs']
verbs = ['walk', 'move', 'swim', 'growl']
#verbs = ['walk']
dets = ['all', 'not_all', 'some', 'no']
#dets = ['all', 'not_all', 'some', 'no', 'two', 'lt_two', 'three', 'lt_three']
#dets = ['all']
adverbs = ['', 'not']

#conflicts_counter = Counter()

#TODO TAKE INTO CONSIDERATION THAT 'UNKNOWN' RELATION IS MOSTLY IGNORED IN NL

def conflicting_sentences(ignore_unk=True):

    # Uncomment to randomize order of sentence combinations (optional)
    all_combinations = list(product(dets, dets, adverbs, adverbs, nouns, nouns, adverbs, adverbs, verbs, verbs))
    random.shuffle(all_combinations)

    # Randomly sample n combinations => max n conflicts
    sample_combinations = all_combinations[:SAMPLE_SIZE]

    # Only take quantifiers from FOL because these form a subset of the NL ones (without most/not_most)
    #for d1, d2, na1, na2, n1, n2, va1, va2, v1, v2 in product(dets, dets, adverbs, adverbs, nouns, nouns, adverbs, adverbs, verbs, verbs):
    for d1, d2, na1, na2, n1, n2, va1, va2, v1, v2 in sample_combinations:

        d = {}
        s = [[(d1, d2), [(na1, na2), (n1, n2)]], [(va1, va2), (v1, v2)]]
        d['sentence'] = s
        d['premise'] = fol.leaves(s, 0)
        d['hypothesis'] = fol.leaves(s, 1)

        # Relation according to natural logic calculus (Bowman)
        d['relation_nl'] = nl.interpret(s, nl.lexicon, nl.projectivity)[0]

        # Filter axioms to reduce recursive depth of proof
        filtered_axioms = fol.filter_axioms(fol.axioms, d1, d2, na1, na2, n1, n2, v1, v2)

        # Relation according to first order logic calculus
        d['relation_fol'] = fol.interpret(s, filtered_axioms)

        if d['relation_nl'] != d['relation_fol']:
            #conflicts_counter[d['relation_nl'] + d['relation_fol']] += 1
            yield d

def matlab_string(d):
    return str(d['relation_nl']) + '\t' + str(d['relation_fol']) + '\t' + str(nl.sentence_to_parse(d['premise'])) + '\t' + str(nl.sentence_to_parse(d['hypothesis']))

def visualize(counters):
    """

    :param counters: list of Counter() objects
    :return:
    """

    cumulative_counts = sum(counters, Counter())

    # average wrt number of samples
    for key in cumulative_counts:
        cumulative_counts[key] /= SAMPLES

    # plot

    labels, values = zip(*cumulative_counts.items())
    sorted_values = sorted(values)[::-1]
    sorted_labels = [x for (y,x) in sorted(zip(values,labels))][::-1]
    indexes = np.arange(len(sorted_labels))
    width = 1

    plt.bar(indexes, sorted_values)
    plt.xticks(indexes + width * 0.5, sorted_labels)

    plt.title("Average frequencies of NL vs FOL conflicts wrt " + str(SAMPLES) + " samples of size " + str(SAMPLE_SIZE))
    plt.xlabel("Conflict")
    plt.ylabel("Avg frequency")
    plt.savefig("avg_freq_conflicts_" + str(SAMPLES) + "samples_" + str(SAMPLE_SIZE) + "size")
    #plt.show()
    plt.close()

    # plot again but take out first one (?#) because it is so much bigger than the rest

    plt.bar(indexes[1:], sorted_values[1:])
    plt.xticks(indexes[1:] + width * 0.5, sorted_labels[1:])

    plt.title("Average frequencies of NL vs FOL conflicts wrt " + str(SAMPLES) + " samples of size " + str(SAMPLE_SIZE) + " without ? vs #")
    plt.xlabel("Conflict")
    plt.ylabel("Avg frequency")
    plt.savefig("avg_freq_conflicts_" + str(SAMPLES) + "samples_" + str(SAMPLE_SIZE) + "size_without?#")
    #plt.show()


if __name__ == '__main__':
    #conflicts_counter = Counter()
    counters = []

    if FILE_OUTPUT:

        output_file = open(FILENAME_STEM + "output.txt", 'w')

        for i in range(SAMPLES):
            conflicts_counter = Counter()
            output_file.write('SAMPLE ' + str(i) + '\n')

            counters = {}

            conflicts = conflicting_sentences()

            sentences = set()
            for counter, d in enumerate(conflicts):
                sentences.add(tuple(d['premise']))
                sentences.add(tuple(d['hypothesis']))
                conflicts_counter[d['relation_nl'] + d['relation_fol']] += 1

            sentence_list = list(sentences)

            for counter, d in enumerate(conflicting_sentences()):
                output_file.write(matlab_string(d) + "\n")

            for conflict, frequency in conflicts_counter.most_common():
                output_file.write("NL %s vs. FOL %s present %s times \n" %(conflict[0], conflict[1], frequency))

            output_file.write("\n")

            counters.append(conflicts_counter)

        output_file.close()

    else:
        for i in range(SAMPLES):
            conflicts_counter = Counter()
            print('SAMPLE ' + str(i))
            for counter, d in enumerate(conflicting_sentences()):
                print("======================================================================")
                print('Sentence %s:' % counter, d['sentence'])
                print('Premise:    ', d['premise'])
                print('Hypothesis: ', d['hypothesis'])
                print('NL Relation:   ', d['relation_nl'])
                print('FOL Relation:   ', d['relation_fol'])
                conflicts_counter[d['relation_nl'] + d['relation_fol']] += 1

            for conflict, frequency in conflicts_counter.most_common():
                print("NL %s vs. FOL %s present %s times \n" %(conflict[0], conflict[1], frequency))

            counters.append(conflicts_counter)

            print('\n')

    if VISUALIZE:
        visualize(counters)

