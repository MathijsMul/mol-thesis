"""
generates more complex FOL data
"""

import logging
import datetime
import fol_lexicon as fol_lex
from itertools import product
from operator import itemgetter
from collections import defaultdict, Counter
import random
import sys

from nltk.inference import Prover9, Mace
from nltk.sem import Expression

logging.getLogger()
logging.basicConfig(format='%(message)s', level=logging.INFO)

read_expr = Expression.fromstring

INDY_DOWNSAMPLE_RATIO = 0.05
FILE_OUTPUT = False
PROVER_ON = True # set to False in case we just want to list sentence combinations without running the theorem prover
#FILENAME_STEM = "binary_neg_noun2"

SAMPLE_DATA = True
if SAMPLE_DATA:
    #sample_probability = 0.01
    sample_probability = 0.025 # take this one for final data
else:
    sample_probability = 1.00

# initialize theorem prover and model builder
prover = Prover9(timeout=1)
mace = Mace(end_size=4)
#mace.config_prover9(r'/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/generate_data/LADR-2009-11A/bin')

TAXONOMY = "people_binary_decompquant"
dets, adverbs, nouns, verbs, noun_matrix, verb_matrix = fol_lex.get_taxonomy(TAXONOMY)
fol_lexicon = fol_lex.get_lexicon(nouns, verbs, noun_matrix, verb_matrix)

adverbs_det1 = adverbs
adverbs_noun1 = adverbs
adverbs_verb = adverbs
adverbs_det2 = ['']
adverbs_noun2 = adverbs

def leaves(s, dim):
    """
    from Bowman, for visualizing an aligned tree s. dim=0 for premise; dim=1 for hypothesis
    """

    l = []
    for x in s:
        if isinstance(x, tuple):
            l += [x[dim]]
        else:
            l += leaves(x, dim)
    return l

def general_axioms(lexicon):
    """
    generate total list of axioms

    :param lexicon: lexical relations
    :return: list of axioms
    """

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
                axiom_strings += ['all x. all y. ({}(x, y) -> {}(x, y))'.format(pair[0], pair[1])]

            # reverse entailment
            elif relation == '>':
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

    # remove duplicates
    for term, axiom_dict in axioms.items():
        for term, axiom_list in axiom_dict.items():
            axiom_list = list(set(axiom_list))

    return(axioms)

axioms = general_axioms(fol_lexicon)

def sentence_to_fol(sentence):
    """
    formalizes list of sentence compounds as recognizable FOL wff
    """

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

    elif det_subj == 'two':
        parse += 'exists x. exists y. ( ' + subj + '(x) and ' + subj + '(y) and x != y and '

        if adv_verb == 'not':
            parse += 'not '

        if adv_det_obj == 'not':
            parse += 'not'

        if det_obj == 'some':
            parse += 'exists v. (' + obj + '(v) and ' + verb + '(x,v) and ' + verb + '(y,v)))'

        elif det_obj == 'all':
            parse += 'all v. (' + obj + '(v) -> ' + verb + '(x,v) and ' + verb + '(y,v)))'

        elif det_obj == 'two':
            parse += 'exists v. exists w. (' + obj + '(v) and ' + obj + '(w) and v != w and ' + verb + '(x,v) and ' + verb + '(y,v) and ' + verb + '(x,w) and ' + verb + '(y,w)))'
        return(parse)

    if adv_verb == 'not':
        parse += 'not '

    if adv_det_obj == 'not':
        parse += 'not'

    if det_obj == 'some':
        parse += 'exists y. (' + obj + '(y) and ' + verb + '(x,y)))'

    elif det_obj == 'all':
        parse += 'all y. (' + obj + '(y) -> ' + verb + '(x,y)))'

    elif det_obj == 'two':
        parse += 'exists y. exists z. (' + obj + '(y) and ' + obj + '(z) and y != z and ' + verb + '(x,y) and ' + verb + '(x,z)))'

    return(parse)

def interpret(sentence, axioms):
    """
    determine relation between sentences
    """

    left_fol = sentence_to_fol(leaves(sentence, 0))
    right_fol = sentence_to_fol(leaves(sentence, 1))
    left = read_expr(left_fol)
    not_left = read_expr('not ' + left_fol)
    right = read_expr(right_fol)

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
    """
    select only relevant axioms, do not consider ones without currently occurring verbs/nouns
    """

    noun_axioms = axioms[noun_subj1][noun_subj2] + axioms[noun_subj2][noun_subj1] \
                  + axioms[noun_obj1][noun_obj2] + axioms[noun_obj2][noun_obj1]
    verb_axioms = axioms[verb1][verb2] + axioms[verb2][verb1]

    existence_axioms = []

    # existential import axioms
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
    """
    generator for all sentences
    """

    for da_subj, det_subj, na_subj, n_subj, va_subj, v_subj, da_obj, det_obj, va_obj, n_obj \
            in product(adverbs_det1, dets, adverbs_noun1, nouns, adverbs_verb, verbs, adverbs_det2, dets, adverbs_noun2, nouns):
        sentence = tuple([da_subj, det_subj, na_subj, n_subj, va_subj, v_subj, da_obj, det_obj, va_obj, n_obj])

        yield sentence

def all_pairs(ignore_unk=True):
    """
    generator for the current grammar and lexicon
    (can be made faster by not considering sentence pairs without shared terms, as these are # anyway)
    """

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

            # filter axioms to reduce recursive depth of proof
            filtered_axioms = filter_axioms(axioms, det_subj1, det_subj2, na_subj1, na_subj2, \
                                            n_subj1, n_subj2, v_subj1, v_subj2, n_obj1, n_obj2)
            if PROVER_ON:
                d['relation'] = interpret(s, filtered_axioms)
            yield d

def sentence_to_parse(sentence):
    """
    add brackets according to syntactic structure
    """

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
        parse += '( ( not '+ det_obj + ' ) '
    else:
        parse += '( ' + det_obj + ' '

    if adv_obj == 'not':
        parse += '( not ' + noun_obj + ' ) '
    else:
        parse += noun_obj + ' '
    parse += ') ) ) '

    return parse

def file_string(d):
    return str(d['relation']) + '\t' + str(sentence_to_parse(d['premise'])) + '\t' + str(sentence_to_parse(d['hypothesis']))

if __name__ == '__main__':
    FILENAME_STEM = sys.argv[1]

    start = datetime.datetime.now()
    logging.info("Start time: %s" % start)

    if FILE_OUTPUT:
        bulk_file = open(FILENAME_STEM + 'bulk.txt', 'w')

        for counter, d in enumerate(all_pairs()):
            if counter % 100 == 0:
                print('Analyzing pair %d' % counter)
            bulk_file.write(file_string(d) + '\n')

        bulk_file.close()

    else:
        for counter, d in enumerate(all_pairs()):
            print("======================================================================")
            print('Sentence %s:' % counter, d['sentence'])
            print('Premise:    ', d['premise'])
            print('Hypothesis: ', d['hypothesis'])

            if PROVER_ON:
                print('Relation:   ', d['relation'])

    logging.info("Start time: %s" % start)
    logging.info("End time: %s" % datetime.datetime.now())
