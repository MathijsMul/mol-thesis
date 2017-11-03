# generates more complex FOL data with following properties:
# - no taxonomic symmetries
# - decomposed negated quantifiers
# - binary predicates

import logging
import datetime

import fol_lexicon as fol_lex

from itertools import product
from operator import itemgetter
from collections import defaultdict
from collections import Counter
import random

from nltk.inference import Prover9
from nltk.inference import Mace
from nltk.sem import Expression

logging.getLogger()
logging.basicConfig(format='%(message)s', level=logging.INFO)

read_expr = Expression.fromstring

INDY_DOWNSAMPLE_RATIO = 0.05
MATLAB_OUTPUT = True
PROVER_ON = True # set to False in case we just want to list sentence combinations without running the theorem prover
FILENAME_STEM = "binary_neg_noun1"

SAMPLE_DATA = False
if SAMPLE_DATA:
    sample_probability = 0.0001
else:
    sample_probability = 1.00

# Currently Prover9, first used TableauProver, but that one could not handle any sentences with quantifiers
# three or lt_three. Prover9 still gives problems with these, though.
# timeout set to 1 second, because for higher running times no solution is found anyway => switch to Mace4 model builder
# in such cases
prover = Prover9(timeout=1)

# model builder Mace, checks for satisfying models with max domain size 10. actually this could still be decreased.
mace = Mace(end_size=10)

TAXONOMY = "people_binary_decompquant"
dets, adverbs, nouns, verbs, noun_matrix, verb_matrix = fol_lex.get_taxonomy(TAXONOMY)
fol_lexicon = fol_lex.get_lexicon(nouns, verbs, noun_matrix, verb_matrix)

# do not include negation at all possible locations, because otherwise the number of sentences explodes
# preferably, negation is allowed at one point in the sentence. this can be varied to study whether the
# learned representation is similar for negated quantifiers/nouns/verbs
adverbs_det1 = ['']
adverbs_noun1 = ['']
adverbs_verb = adverbs # seems to have same function as adverbs_det2, so they can cancel each other out
adverbs_det2 = ['']
adverbs_noun2 = ['']

def leaves(s, dim):
    """For visualizing an aligned tree s. dim=0 for premise; dim=1 for hypothesis."""
    l = []
    for x in s:
        if isinstance(x, tuple):
            l += [x[dim]]
        else:
            l += leaves(x, dim)
    return l

def general_axioms(lexicon):
    #print(lexicon)
    axioms = {}
    for noun1 in nouns:
        axioms[noun1] = {}
        for noun2 in nouns:
            axioms[noun1][noun2] = []
    for verb1 in verbs:
        axioms[verb1] = {}
        for verb2 in verbs:
            axioms[verb1][verb2] = []

    for pair, relation in lexicon.items():
        axiom_strings = []

        if pair[0] in verbs and pair[1] in verbs:
            # binary

            # forward entailment
            if relation == '<':
                #axiom_strings += ['all x.({}(x) -> {}(x))'.format(pair[0], pair[1])]
                axiom_strings += ['all x. all y. ({}(x, y) -> {}(x, y))'.format(pair[0], pair[1])]

            # reverse entailment
            elif relation == '>':
                #axiom_strings = ['all x.({}(x) -> {}(x))'.format(pair[1], pair[0])]
                axiom_strings += ['all x. all y. ({}(x, y) -> {}(x, y))'.format(pair[1], pair[0])]

            # alternation
            elif relation == "|":
                axiom_strings += ['all x. all y. (not ({}(x, y) and {}(x, y)) )'.format(pair[0], pair[1])]
                axiom_strings += ['not (all x. all y. ({}(x, y) or {}(x, y)))'.format(pair[0], pair[1])]

            # negation
            elif relation == "^":
                axiom_strings += ['all x. all y. (not ({}(x, y) and {}(x, y)) )'.format(pair[0], pair[1])]
                axiom_strings += ['all x. all y. ({}(x, y) or {}(x, y))'.format(pair[0], pair[1])]

            # cover
            elif relation ==  "v":
                axiom_strings += ['all x. all y. ((not {}(x, y)) -> {}(x, y) )'.format(pair[0], pair[1])]

            for axiom_string in axiom_strings:
                #print(axiom_string)
                axioms[pair[0]][pair[1]] += [read_expr(axiom_string)]

        else:
            # unary

            # forward entailment
            if relation == '<':
                axiom_strings += ['all x.({}(x) -> {}(x))'.format(pair[0], pair[1])]

            # reverse entailment
            elif relation == '>':
                axiom_strings = ['all x.({}(x) -> {}(x))'.format(pair[1], pair[0])]

            # alternation
            elif relation == "|":
                axiom_strings += ['all x.(not ({}(x) and {}(x)) )'.format(pair[0], pair[1])]
                axiom_strings += ['not all x.({}(x) or {}(x))'.format(pair[0], pair[1])]

            # negation
            elif relation == "^":
                axiom_strings += ['all x.(not ({}(x) and {}(x)) )'.format(pair[0], pair[1])]
                axiom_strings += ['all x.({}(x) or {}(x))'.format(pair[0], pair[1])]

            # cover
            elif relation ==  "v":
                axiom_strings += ['all x.((not {}(x)) -> {}(x) )'.format(pair[0], pair[1])]

            for axiom_string in axiom_strings:
                #print(axiom_string)
                axioms[pair[0]][pair[1]] += [read_expr(axiom_string)]

    # Remove duplicates
    for term, axiom_dict in axioms.items():
        for term, axiom_list in axiom_dict.items():
            axiom_list = list(set(axiom_list))

    return(axioms)

axioms = general_axioms(fol_lexicon)

def sentence_to_fol(sentence):
    # TODO: add numeric quantifiers?


    adv_det_subj = sentence[0]
    det_subj = sentence[1]
    adv_subj = sentence[2]
    noun_subj = sentence[3]
    adv_verb = sentence[4]
    verb = sentence[5]
    adv_det_obj = sentence[6]
    det_obj = sentence[7]
    adv_obj = sentence[8]
    noun_obj = sentence[9]

    if adv_subj == 'not':
        subj = 'not ' + noun_subj
    else:
        subj = noun_subj

    if adv_obj == 'not':
        obj = 'not ' + noun_obj
    else:
        obj = noun_obj

    if adv_det_subj == 'not':
        parse = 'not '
    else:
        parse = ''

    if det_subj == 'some':
        parse += 'exists x. ( ' + subj + '(x) and '

    elif det_subj == 'all':
        parse += 'all x. ( ' + subj + '(x) ->'

    if adv_verb == 'not':
            parse += 'not '

    # so negation of second quantifier receives the same interpretation as negation of verb => check this
    if adv_det_obj == 'not':
        parse += 'not'

    if det_obj == 'some':
        parse += 'exists y. (' + obj + '(y) and ' + verb + '(x,y)))'

    elif det_obj == 'all':
        parse += 'all y. (' + obj + '(y) -> ' + verb + '(x,y)))'
    #print(parse)
    return(parse)

def interpret(sentence, axioms):
    """Determine relation between sentences"""

    left_fol = sentence_to_fol(leaves(sentence, 0))
    right_fol = sentence_to_fol(leaves(sentence, 1))
    # print(left_fol)
    # print(right_fol)

    left = read_expr(left_fol)
    not_left = read_expr('not ' + left_fol)
    right = read_expr(right_fol)
    # print(left)
    # print(right)

    contradiction = read_expr('not (' + left_fol + ' and ' + right_fol + ')')

    try:
        forward_entailment = prover.prove(right, axioms + [left])
    except:
        forward_entailment = not mace.build_model(right, axioms + [left])

    try:
        backward_entailment = prover.prove(left, axioms + [right])
    except:
        backward_entailment = not mace.build_model(left, axioms + [right])

    if forward_entailment and backward_entailment:
        return('=')
    elif forward_entailment:
        return('<')
    elif backward_entailment:
        return('>')

    try:
        conflict = prover.prove(contradiction, axioms)
    except:
        conflict = not mace.build_model(contradiction, axioms)

    try:
        exhaustion = prover.prove(right, axioms + [not_left])
    except:
        exhaustion = not mace.build_model(right, axioms + [not_left])

    if conflict and not exhaustion:
        return("|")
    elif conflict and exhaustion:
        return("^")
    elif exhaustion:
        return("v")

    else:
        return("#")

def filter_axioms(axioms, det1, det2, adverb1, adverb2, noun_subj1, noun_subj2, verb1, verb2, noun_obj1, noun_obj2):
    """Select only relevant axioms, do not consider ones without currently occurring verbs/nouns"""

    noun_axioms = axioms[noun_subj1][noun_subj2] + axioms[noun_subj2][noun_subj1] \
                  + axioms[noun_obj1][noun_obj2] + axioms[noun_obj2][noun_obj1]
    verb_axioms = axioms[verb1][verb2] + axioms[verb2][verb1]

    existence_axioms = []

    # Existential import axioms if necessary (not for existential statements)
    # take care of negated terms
    if det1 in ['all', 'not_all']:
        if adverb1 == '':
            existence_axioms += [read_expr('exists x.({}(x))'.format(noun_subj1))]

        elif adverb1 == 'not':
            # negative existential import
            existence_axioms += [read_expr('exists x.(not {}(x))'.format(noun_subj1))]

    if det2 in ['all', 'not_all']:
        if adverb2 == '':
            existence_axioms += [read_expr('exists x.({}(x))'.format(noun_subj2))]

        elif adverb2 == 'not':
            # negative existential import
            existence_axioms += [read_expr('exists x.(not {}(x))'.format(noun_subj2))]

    relevant_axioms = list(set(noun_axioms + verb_axioms + existence_axioms))

    return relevant_axioms

def all_sentences():
    for da_subj, det_subj, na_subj, n_subj, va_subj, v_subj, da_obj, det_obj, va_obj, n_obj \
            in product(adverbs_det1, dets, adverbs_noun1, nouns, adverbs_verb, verbs, adverbs_det2, dets, adverbs_noun2, nouns):
        sentence = tuple([da_subj, det_subj, na_subj, n_subj, va_subj, v_subj, da_obj, det_obj, va_obj, n_obj])

        yield sentence

def all_pairs(ignore_unk=True):
    """Generator for the current grammar and lexicon. Yields dicts with useful info."""

    # TODO: make this faster by not considering sentence pairs without interpolants (shared terms), as these are # anyway

    #count_control = 0
    #print(len(list(product(adverbs_det1, adverbs_det1, dets, dets, adverbs_noun1, adverbs_noun1, nouns, nouns, adverbs_verb, adverbs_verb, verbs, verbs, dets, dets, adverbs_noun2, adverbs_noun2, nouns, nouns))))

    #for d1, d2, na1, na2, n1, n2, va1, va2, v1, v2 in product(dets, dets, adverbs, adverbs, nouns, nouns, adverbs, adverbs, verbs, verbs):
    for da_subj1, da_subj2, det_subj1, det_subj2, na_subj1, na_subj2, n_subj1, n_subj2, \
        va_subj1, va_subj2, v_subj1, v_subj2, da_obj1, da_obj2, det_obj1, det_obj2, va_obj1, va_obj2, n_obj1, n_obj2 \
        in product(adverbs_det1, adverbs_det1, dets, dets, adverbs_noun1, adverbs_noun1, nouns, nouns, adverbs_verb, adverbs_verb, verbs, verbs, \
                   adverbs_det2, adverbs_det2, dets, dets, adverbs_noun2, adverbs_noun2, nouns, nouns ):

        if random.random() < sample_probability:
            d = {}
            s = [[[(da_subj1, da_subj2), (det_subj1, det_subj2)], [(na_subj1, na_subj2), (n_subj1, n_subj2)]], \
                 [[(va_subj1, va_subj2), (v_subj1, v_subj2)], [[(da_obj1, da_obj2), (det_obj1, det_obj2)], [(va_obj1, va_obj2), (n_obj1, n_obj2)]]]]

            d['sentence'] = s
            d['premise'] = leaves(s, 0)
            d['hypothesis'] = leaves(s, 1)
            #print(d)

            # TODO: filter axioms according to new sentence structure
            # Filter axioms to reduce recursive depth of proof
            filtered_axioms = filter_axioms(axioms, det_subj1, det_subj2, na_subj1, na_subj2, \
                                            n_subj1, n_subj2, v_subj1, v_subj2, n_obj1, n_obj2)

            # print('filtered axioms:')
            # print(filtered_axioms)

            if PROVER_ON:
                d['relation'] = interpret(s, filtered_axioms)

            # TODO: how to downsample #? probabilistically, or somewhow check wrt NL?
            # we must normalize wrt # labels, because otherwise they would completely dominate the data set and make its size explode.
            # two options:
            # 1. only accept pair labeled with # if if it is also labeled # in NL (and not ?) (time consuming)
            # 2. only accept label # with probability 1/p, where p ~ 10
            # if d['relation'] == "#":
            #     if nl.interpret(s, nl_lexicon, nl.projectivity)[0] != "?": # option 1
            #         yield d
#           else:
#               yield d

            #if (d['relation'] != '#') or (d['relation'] == '#' and random.random() < INDY_DOWNSAMPLE_RATIO):
            yield d

def label_distribution():
    """Calculates the distribution of labels for the current grammar."""
    counts = defaultdict(int)
    for d in all_pairs():
        counts[d['relation']] += 1
    total = float(sum(counts.values()))
    for key, val in sorted(counts.items(), key=itemgetter(1), reverse=True):
        print(key, val, val / total)

def sentence_to_parse(sentence):

    adv_det_subj = sentence[0]
    det_subj = sentence[1]
    adv_subj = sentence[2]
    noun_subj = sentence[3]
    adv_verb = sentence[4]
    verb = sentence[5]
    adv_det_obj = sentence[6]
    det_obj = sentence[7]
    adv_obj = sentence[8]
    noun_obj = sentence[9]

    parse = ' ( ( '

    if adv_det_subj == 'not':
        parse += '( not ' + det_subj + ' ) '
    else:
        parse += det_subj + ' '

    if adv_subj == 'not':
        parse += '( not ' + noun_subj + ' ) '
    else:
        parse += noun_subj + ' '

    parse += ' ) ( '

    if adv_verb == 'not':
        parse += '( not ' + verb + ' ) '
    else:
        parse += verb + ' '

    if adv_det_obj == 'not':
        parse += '( ( not  '+ det_obj + ' ) '
    else:
        parse += '( ' + det_obj + ' '

    if adv_obj == 'not':
        parse += '( ' + noun_obj + ' ) '
    else:
        parse += noun_obj + ' '
    parse += ') ) ) '

    return parse


def matlab_string(d):
    return str(d['relation']) + '\t' + str(sentence_to_parse(d['premise'])) + '\t' + str(sentence_to_parse(d['hypothesis']))

if __name__ == '__main__':

    logging.info("Start time: %s" % datetime.datetime.now())

    if MATLAB_OUTPUT:

        bulk_file = open(FILENAME_STEM + 'bulk.txt', 'w')

        for counter, d in enumerate(all_pairs()):
            #print(d)
            #print(counter)
            if counter % 100 == 0:
                print('Analyzing pair %d' % counter)
            #print(matlab_string(d))
            bulk_file.write(matlab_string(d) + '\n')
        #
        # training_file = open(FILENAME_STEM + "train.txt", 'w')
        # test_file = open(FILENAME_STEM + "test.txt", 'w')
        #
        # counters = {}
        #
        # sentences = set() # is set even necessary here? there won't be duplicates anyway in the cartesian product
        # for counter, s in enumerate(all_sentences()):
        #     if counter % 100 == 0:
        #         print('Reading sentence %d' % counter)
        #     sentences.add(s)
        #
        # sentence_list = list(sentences)
        # random.shuffle(sentence_list)
        #
        # test_examples = sentence_list[1:int(.33 * len(sentence_list))]
        #
        # # Original Bowman code was horribly inefficient for this part. Analyzed all labels twice, but once only to construct
        # # the list of single sentences.
        # for counter, d in enumerate(all_pairs()):
        #     #print(counter)
        #     if counter % 100 == 0:
        #         print('Analyzing pair %d' % counter)
        #
        #     if tuple(d['premise']) in test_examples and tuple(d['hypothesis']) in test_examples:
        #         test_file.write(matlab_string(d) + "\n")
        #     elif not (tuple(d['premise']) in test_examples or tuple(d['hypothesis']) in test_examples):
        #         training_file.write(matlab_string(d) + "\n")
        #
        # training_file.close()
        # test_file.close()

        # print('total # pairs:')
        # print(counter)
        bulk_file.close()

    else:
        for counter, d in enumerate(all_pairs()):
            print("======================================================================")
            print('Sentence %s:' % counter, d['sentence'])
            print('Premise:    ', d['premise'])
            print('Hypothesis: ', d['hypothesis'])

            if PROVER_ON:
                print('Relation:   ', d['relation'])

            #print('\n')

            # if counter == 3:
            #     break

    logging.info("End time: %s" % datetime.datetime.now())
