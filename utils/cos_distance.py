"""
compute cosine distance between word embeddings
"""

import numpy as np
import scipy.spatial.distance as sp

glove_path = '/Users/mathijs/Documents/Studie/MoL/thesis/glove.6B/glove.6B.50d.txt'

def get_cos_distance(word1, word2):
    """
    compute cosine distance between word embeddings

    :param word1: some word
    :param word2: other word
    :return: cosine distance between input words
    """
    f = open(glove_path, 'r', encoding='utf8')
    embeddings = []
    count = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        if word in [word1.lower(), word2.lower()]:
            embeddings += [np.array([float(val) for val in splitLine[1:]])]
            count += 1
            if count == 2:
                cos_distance = sp.cosine(embeddings[0], embeddings[1])
                #return(cos_distance)
                return("%.2f" % cos_distance)

print(get_cos_distance('Romans', 'Venetians'))
