from itertools import product

FOR = "<"
REV = ">"
NEG = "^"
ALT = "|"
COV = "v"
EQ = "="
INDY = "#"

dets = ['all', 'not_all', 'some', 'no', 'two', 'lt_two', 'three', 'lt_three']
#decomp_dets = ['all', 'some', 'two']
decomp_dets = ['all', 'some']


nl_dets = ['all', 'not_all', 'some', 'no', 'most',
        'not_most', 'two', 'lt_two', 'three', 'lt_three']
nl_det_matrix = [
    # all   not_all some    no      most    not_most two    lt_two  three
    # lt_three
    [EQ,	NEG,	FOR,	ALT,	FOR,	ALT,	 INDY,	INDY,	INDY,	INDY],  # all
    [NEG,	EQ,	    COV,	REV,	COV,	REV,	 INDY,	INDY,	INDY,	INDY],  # not_all
    [REV,	COV,	EQ,	    NEG,	REV,	COV,	 REV,	COV,	REV,	COV],  # some
    [ALT,	FOR,	NEG,	EQ,	    ALT,	FOR,	 ALT,	FOR,	ALT,	FOR],  # no
    [REV,	COV,	FOR,	ALT,	EQ,	    NEG,	 INDY,	INDY,	INDY,	INDY],  # most
    [ALT,	FOR,	COV,	REV,	NEG,	EQ,	     INDY,	INDY,	INDY,	INDY],  # not_most
    [INDY,	INDY,	FOR,	ALT,	INDY,	INDY,	 EQ,    NEG,	REV,	COV],  # two
    [INDY,	INDY,	COV,	REV,	INDY,	INDY,	 NEG,	EQ,	    ALT,	FOR],  # lt_two
    [INDY,	INDY,	FOR,	ALT,	INDY,	INDY,	 FOR,	ALT,	EQ,	    NEG],  # three
    [INDY,	INDY,	COV,	REV,	INDY,	INDY,	 COV,	REV,	NEG,	EQ]    # lt_three
]

adverbs = ['', 'not']

# ANIMALS

nouns_animals = ['warthogs', 'turtles', 'mammals', 'reptiles', 'pets']
noun_matrix_animals = [
    # warthogs turtles mammals reptiles pets
    [EQ,       ALT,    FOR,    ALT,     INDY],  # warthogs
    [ALT,      EQ,     ALT,    FOR,     INDY],  # turtles
    [REV,      ALT,    EQ,     ALT,     INDY],  # mammals
    [ALT,      REV,    ALT,    EQ,      INDY],  # reptiles
    [INDY,     INDY,   INDY,   INDY,    EQ]    # pets
]

verbs_animals = ['walk', 'move', 'swim', 'growl']
verb_matrix_animals = [
    # walk move  swim growl
    [EQ,   FOR,  ALT, INDY],  # walk
    [REV,  EQ,   REV, INDY],  # move
    [ALT,  FOR,  EQ,  ALT],  # swim
    [INDY, INDY, ALT, EQ]    # growl
]

# people

nouns_people = ['Romans', 'Italians', 'Europeans', 'Germans', 'children']
#nouns_people = ['Romans', 'Italians']
noun_matrix_people = [
    # romans italians europeans germans children
    [EQ,       FOR,    FOR,    ALT,     INDY],  # romans
    [REV,      EQ,     FOR,    ALT,     INDY],  # italians
    [REV,      REV,    EQ,     REV,     INDY],  # europeans
    [ALT,      ALT,    FOR,    EQ,      INDY],  # germans
    [INDY,     INDY,   INDY,   INDY,    EQ]    # children
]

verbs_people = ['clean', 'work', 'teach', 'laugh']
verb_matrix_people = [
    # clean work teach laugh
    [EQ,   FOR,  ALT, INDY],  # clean
    [REV,  EQ,   REV, INDY],  # work
    [ALT,  FOR,  EQ,  ALT],  # teach
    [INDY, INDY, ALT, EQ]    # laugh
]

verbs_binary_people = ['fear', 'hate', 'like', 'love']
verbs_binary_matrix_people = [
    #fear hate like love
    [EQ,   INDY,  ALT, ALT],  # fear
    [INDY,  EQ,   ALT, ALT],  # hate
    [ALT,  ALT,  EQ,  REV],  # like
    [ALT, ALT, FOR, EQ]    # love
]

def get_taxonomy(name):
    """

    :param name:
    :return: determiners, adverbs, nouns, verbs
    """

    if name == 'animals_test':
        return(dets, adverbs, nouns_animals[0:1], verbs_animals[0:1], noun_matrix_animals, verb_matrix_animals)
    if name == 'animals':
        return(dets, adverbs, nouns_animals, verbs_animals, noun_matrix_animals, verb_matrix_animals)
    if name == 'people':
        return(dets, adverbs, nouns_people, verbs_people, noun_matrix_people, verb_matrix_people)
    if name == 'people_binary_decompquant':
        return(decomp_dets, adverbs, nouns_people, verbs_binary_people, noun_matrix_people, verbs_binary_matrix_people)

def get_lexicon(nouns, verbs, noun_matrix, verb_matrix):

    # New lexicon only contains information for nouns and verbs
    lexicon = {}

    # Only take upper half of matrices, because lower half is implied and we don't want equivalent axioms
    for i, j in product(range(len(nouns)), range(len(nouns))):
        if i <= j:
            lexicon[(nouns[i], nouns[j])] = noun_matrix[i][j]

    for i, j in product(range(len(verbs)), range(len(verbs))):
        if i <= j:
            lexicon[(verbs[i], verbs[j])] = verb_matrix[i][j]

    return lexicon

