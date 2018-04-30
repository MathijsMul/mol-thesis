"""
Code for generating simple quantified statements and calculating
their FOL relation.
"""
import nl_gen as nl
import fol_lexicon as fol_lex
import nl_lexicon as nl_lex

from itertools import product
from operator import itemgetter
from collections import defaultdict
from collections import Counter
import random

from nltk.inference import Prover9
from nltk.inference import Mace
from nltk.sem import Expression

read_expr = Expression.fromstring

# initialize theorem prover and model builder
prover = Prover9(timeout=1)
mace = Mace(end_size=10)

INDY_DOWNSAMPLE_RATIO = 0.05
FILE_OUTPUT = False
PROVER_ON = True # set to False in case we just want to list sentence combinations without running the theorem prover
FILENAME_STEM = "nothing"

SAMPLE_DATA = False
if SAMPLE_DATA:
    sample_probability = 0.03
else:
    sample_probability = 1.00

TAXONOMY = "animals"
dets, adverbs, nouns, verbs, noun_matrix, verb_matrix = fol_lex.get_taxonomy(TAXONOMY)
fol_lexicon = fol_lex.get_lexicon(nouns, verbs, noun_matrix, verb_matrix)

nl_dets, nl_det_matrix = nl_lex.dets, nl_lex.det_matrix
nl_lexicon = nl_lex.get_lexicon(nouns, verbs, nl_dets, noun_matrix, verb_matrix, nl_det_matrix)

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

        # equivalence is redundant, because we only have self-equivalence of primitives

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

    det = sentence[0]

    if det in ['all', 'not_all', 'some', 'no']:
        if det in ['all', 'not_all']:
            parse_start = 'all x.('
        else:
            parse_start = 'exists x.('

        if sentence[1] == 'not':
            parse_mid = '( not ' + sentence[2] + '(x))'
        else:
            parse_mid = sentence[2] + '(x)'

        if sentence[3] == 'not':
            parse_end = '( not ' + sentence[4] + '(x))'
        else:
            parse_end = sentence[4] + '(x)'

    elif det in ['two', 'lt_two']:
        parse_start = 'exists x. exists y. ('

        if sentence[1] == 'not':
            parse_mid = '( not ' + sentence[2] + '(x)) and ( not ' + sentence[2] + '(y))'
        else:
            parse_mid = sentence[2] + '(x) and ' + sentence[2] + '(y)'

        if sentence[3] == 'not':
            parse_end = '( not ' + sentence[4] + '(x)) and ( not ' + sentence[4] + '(y))'
        else:
            parse_end = sentence[4] + '(x) and ' + sentence[4] + '(y)'
        parse_end += 'and x != y'

    elif det in ['three', 'lt_three']:
        parse_start = 'exists x. exists y. exists z. ('

        if sentence[1] == 'not':
            parse_mid = '( not ' + sentence[2] + '(x)) and ( not ' + sentence[2] + '(y)) and ( not ' + sentence[2] + '(z))'
        else:
            parse_mid = sentence[2] + '(x) and ' + sentence[2] + '(y) and ' + sentence[2] + '(z)'

        if sentence[3] == 'not':
            parse_end = '( not ' + sentence[4] + '(x)) and ( not ' + sentence[4] + '(y))  and ( not ' + sentence[4] + '(z))'
        else:
            parse_end = sentence[4] + '(x) and ' + sentence[4] + '(y) and ' + sentence[4] + '(z)'
        parse_end += 'and x != y and x != z and y != z'

    # universal statements take implication
    if det in ['all', 'not_all']:
        parse = parse_start + parse_mid + ' -> ' + parse_end + ')'

    # existential statements take conjunction
    else:
        parse = parse_start + parse_mid + ' and ' + parse_end + ')'

    # negate sentence for negated quantifiers
    if det in ['not_all', 'no', 'lt_two', 'lt_three']:
        parse = 'not (' + parse + ')'

    return parse

def interpret(sentence, axioms):
    """
    determine relation between sentences
    """

    left_fol = sentence_to_fol(leaves(sentence, 0))
    print(left_fol)
    right_fol = sentence_to_fol(leaves(sentence, 1))
    print(right_fol)

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

def filter_axioms(axioms, det1, det2, adverb1, adverb2, noun1, noun2, verb1, verb2):
    """
    select only relevant axioms, do not consider ones without currently occurring verbs/nouns
    """

    noun_axioms = axioms[noun1][noun2] + axioms[noun2][noun1] #+ axioms[noun1][noun1] + axioms[noun2][noun2]
    verb_axioms = axioms[verb1][verb2] + axioms[verb2][verb1]

    existence_axioms = []

    # existential import axioms
    if det1 in ['all', 'not_all']:
        if adverb1 == '':
            existence_axioms += [read_expr('exists x.({}(x))'.format(noun1))]

        elif adverb1 == 'not':
            # negative existential import
            existence_axioms += [read_expr('exists x.(not {}(x))'.format(noun1))]

    if det2 in ['all', 'not_all']:
        if adverb2 == '':
            existence_axioms += [read_expr('exists x.({}(x))'.format(noun2))]

        elif adverb2 == 'not':
            # negative existential import
            existence_axioms += [read_expr('exists x.(not {}(x))'.format(noun2))]

    relevant_axioms = list(set(noun_axioms + verb_axioms + existence_axioms))

    return relevant_axioms

def all_sentences():
    """
    generator for all sentences
    """
    for det, na, n, va, v in product(dets, adverbs, nouns, adverbs, verbs):
        sentence = tuple([det, na, n, va, v])
        yield sentence

def all_pairs(ignore_unk=True):
    """
    generator for the current grammar and lexicon
    """

    for d1, d2, na1, na2, n1, n2, va1, va2, v1, v2 in product(dets, dets, adverbs, adverbs, nouns, nouns, adverbs, adverbs, verbs, verbs):
        if random.random() < sample_probability:
            d = {}
            s = [[(d1, d2), [(na1, na2), (n1, n2)]], [(va1, va2), (v1, v2)]]

            d['sentence'] = s
            d['premise'] = leaves(s, 0)
            d['hypothesis'] = leaves(s, 1)

            # filter axioms to reduce recursive depth of proof
            filtered_axioms = filter_axioms(axioms, d1, d2, na1, na2, n1, n2, v1, v2)

            if PROVER_ON:
                d['relation'] = interpret(s, filtered_axioms)

            if d['relation'] == "#":
                if nl.interpret(s, nl_lexicon, nl.projectivity)[0] != "?":
                    yield d
            else:
                yield d

def sentence_to_parse(sentence):
    """
    add brackets according to syntactic structure
    """

    parse = ' ( ( ' + sentence[0] + ' '
    if sentence[1] == 'not':
        parse = parse + '( ' + sentence[1] + ' ' + sentence[2] + ' ) ) '
    else:
        parse = parse + sentence[2] + ' ) '
    if sentence[3] == 'not':
        parse = parse + '( ' + sentence[3] + ' ' + sentence[4] + ' ) )'
    else:
        parse = parse + sentence[4] + ' )'
    return parse


def file_string(d):
    return str(d['relation']) + '\t' + str(sentence_to_parse(d['premise'])) + '\t' + str(sentence_to_parse(d['hypothesis']))

if __name__ == '__main__':
    if FILE_OUTPUT:
        training_file = open(FILENAME_STEM + "train.txt", 'w')
        test_file = open(FILENAME_STEM + "test.txt", 'w')

        counters = {}

        sentences = set() # is set even necessary here? there won't be duplicates anyway in the cartesian product
        for counter, s in enumerate(all_sentences()):
            if counter % 100 == 0:
                print('Reading sentence %d' % counter)
            sentences.add(s)

        sentence_list = list(sentences)
        random.shuffle(sentence_list)

        test_examples = sentence_list[1:int(.33 * len(sentence_list))]

        for counter, d in enumerate(all_pairs()):
            if counter % 100 == 0:
                print('Analyzing pair %d' % counter)
            if tuple(d['premise']) in test_examples and tuple(d['hypothesis']) in test_examples:
                test_file.write(file_string(d) + "\n")
            elif not (tuple(d['premise']) in test_examples or tuple(d['hypothesis']) in test_examples):
                training_file.write(file_string(d) + "\n")

        training_file.close()
        test_file.close()

    else:
        for counter, d in enumerate(all_pairs()):
            print("======================================================================")
            print('Sentence %s:' % counter, d['sentence'])
            print('Premise:    ', d['premise'])
            print('Hypothesis: ', d['hypothesis'])

            if PROVER_ON:
                print('Relation:   ', d['relation'])
