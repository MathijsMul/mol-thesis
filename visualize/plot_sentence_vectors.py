"""
plot sentence vectors as outputted by different models,
color according to different constituents
"""

import torch
from rnn import RNN
from trntn import tRNTN
import datamanager as dat
from test import compute_accuracy
import random
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# random.seed(9001)

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']

word_dim = 25
n_hidden = 128
cpr_dim = 75

model_type = 'trntn'

if model_type == 'trntn':
    test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
    test_data = dat.SentencePairsDataset(test_data_file)
    test_data.load_data(sequential=False)

    model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol/models/tRNTNbinary_2dets_4negs_train4.pt'
    trntn = tRNTN(vocab, rels, word_dim, cpr_dim, None, None)
    trntn.load_state_dict(torch.load(model_path))


elif model_type == 'gru':
    test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt'
    test_data = dat.SentencePairsDataset(test_data_file)
    test_data.load_data(sequential=True)

    model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs1.pt'
    gru = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
    gru.load_state_dict(torch.load(model_path))

total_correct = 0
total = 0

downsample_ratio = 0.6
sentence_to_idx = defaultdict(lambda: len(sentence_to_idx))
idx_to_sentence = dict()
idx_to_vector = dict()

idx_to_verb = dict()
idx_to_quant1 = dict()
idx_to_quant2 = dict()
idx_to_bothquants = dict()
idx_to_noun1 = dict()
idx_to_noun2 = dict()
idx_to_np2 = dict()
idx_to_negquant1 = dict()
idx_to_negnoun1 = dict()
idx_to_negverb = dict()
idx_to_negnoun2 = dict()
idx_to_verbquant2 = dict()
idx_to_fullverbquant2 = dict()
idx_to_length = dict()

# idx_to_pca_vector = dict()
relation_lines = {}
for rel in rels:
    relation_lines[rel] = []

def determine_verb(sentence_tree):
    if 'like' in sentence_tree:
        return('like')
    elif 'love' in sentence_tree:
        return('love')
    elif 'hate' in sentence_tree:
        return('hate')
    elif 'fear' in sentence_tree:
        return('fear')

def determine_first_quantifier(sentence_tree):
    if sentence_tree[0] == 'not':
        if sentence_tree[1] == 'some':
            return('not_some')
        elif sentence_tree[1] == 'all':
            return('not_all')
    else:
        return(sentence_tree[0])

def determine_second_quantifier(sentence_tree):
    sentence_without_firstquant = sentence_tree[2:]
    if 'some' in sentence_without_firstquant:
        return('some')
    elif 'all' in sentence_without_firstquant:
        return('all')

def determine_second_noun(sentence_tree):
    return(sentence_tree[-1])

def determine_first_noun(sentence_tree):
    sentence_tree = sentence_tree[:-3]
    if 'Romans' in sentence_tree:
        return('Romans')
    elif 'Italians' in sentence_tree:
        return('Italians')
    elif 'Germans' in sentence_tree:
        return('Germans')
    elif 'Europeans' in sentence_tree:
        return('Europeans')
    elif 'children' in sentence_tree:
        return('children')

def determine_last_np(sentence_tree):
    noun2 = sentence_tree[-1]
    if sentence_tree[-2] == 'not':
        return('not' + noun2)
    else:
        return(noun2)

def determine_bothquants(sentence_tree):
    quant1 = determine_first_quantifier(sentence_tree)
    quant2 = determine_second_quantifier(sentence_tree)
    bothquants = quant1 + quant2
    return(bothquants)

def determine_negquant1(sentence_tree):
    if sentence_tree[0] == 'not':
        return('not')
    else:
        return('')

def determine_negnoun1(sentence_tree):
    sentence_tree = sentence_tree[:-3]
    for noun in ['Romans', 'Italians', 'Germans', 'Europeans', 'children']:
        if noun in sentence_tree:
            noun_idx = sentence_tree.index(noun)
            if sentence_tree[noun_idx - 1] == 'not':
                return('not')
            else:
                return('')

def determine_negverb(sentence_tree):
    for verb in ['like', 'love', 'hate', 'fear']:
        if verb in sentence_tree:
            verb_idx = sentence_tree.index(verb)
            if sentence_tree[verb_idx - 1] == 'not':
                return('not')
            else:
                return('')

def determine_negnoun2(sentence_tree):
    noun2 = sentence_tree[-1]
    if sentence_tree[-2] == 'not':
        return('not')
    else:
        return('')

def determine_verbquant2(sentence_tree):
    verb = determine_verb(sentence_tree)
    if verb in ['fear', 'hate']:
        verb = 'fear/hate'
    elif verb in ['like', 'love']:
        verb = 'like/love'
    quant2 = determine_second_quantifier(sentence_tree)
    verbquant2 = verb + quant2
    return(verbquant2)

def determine_fullverbquant2(sentence_tree):
    verb = determine_verb(sentence_tree)
    quant2 = determine_second_quantifier(sentence_tree)
    verbquant2 = verb + quant2
    return(verbquant2)

rel_count = 0
for idx, item in enumerate(test_data.tree_data):
    if random.random() < downsample_ratio:
        instance = dat.SentencePairsDataset(None)
        instance.tree_data = [item]
        if model_type == 'gru':
            correct = compute_accuracy(instance, rels, gru, print_outputs=False, confusion_matrix=False, batch_size=1)
        elif model_type == 'trntn':
            correct = compute_accuracy(instance, rels, trntn, print_outputs=False, confusion_matrix=False, batch_size=1)

        if correct == '100.00':
            relation = item[0]

            if model_type == 'gru':
                left_sentence = ' '.join(item[1])
                right_sentence = ' '.join(item[2])
                idx_to_sentence[sentence_to_idx[left_sentence]] = left_sentence
                idx_to_sentence[sentence_to_idx[right_sentence]] = right_sentence

                idx_to_vector[sentence_to_idx[left_sentence]] = gru.get_sentence_vector([item[1]])[0]
                idx_to_vector[sentence_to_idx[right_sentence]] = gru.get_sentence_vector([item[2]])[0]

                idx_to_verb[sentence_to_idx[left_sentence]] = determine_verb(item[1])
                idx_to_verb[sentence_to_idx[right_sentence]] = determine_verb(item[2])

                idx_to_quant1[sentence_to_idx[left_sentence]] = determine_first_quantifier(item[1])
                idx_to_quant1[sentence_to_idx[right_sentence]] = determine_first_quantifier(item[2])

                idx_to_quant2[sentence_to_idx[left_sentence]] = determine_second_quantifier(item[1])
                idx_to_quant2[sentence_to_idx[right_sentence]] = determine_second_quantifier(item[2])

                idx_to_bothquants[sentence_to_idx[left_sentence]] = determine_bothquants(item[1])
                idx_to_bothquants[sentence_to_idx[right_sentence]] = determine_bothquants(item[2])

                idx_to_noun1[sentence_to_idx[left_sentence]] = determine_first_noun(item[1])
                idx_to_noun1[sentence_to_idx[right_sentence]] = determine_first_noun(item[2])

                idx_to_noun2[sentence_to_idx[left_sentence]] = determine_second_noun(item[1])
                idx_to_noun2[sentence_to_idx[right_sentence]] = determine_second_noun(item[2])

                idx_to_np2[sentence_to_idx[left_sentence]] = determine_last_np(item[1])
                idx_to_np2[sentence_to_idx[right_sentence]] = determine_last_np(item[2])

                idx_to_negquant1[sentence_to_idx[left_sentence]] = determine_negquant1(item[1])
                idx_to_negquant1[sentence_to_idx[right_sentence]] = determine_negquant1(item[2])

                idx_to_negnoun1[sentence_to_idx[left_sentence]] = determine_negnoun1(item[1])
                idx_to_negnoun1[sentence_to_idx[right_sentence]] = determine_negnoun1(item[2])

                idx_to_negverb[sentence_to_idx[left_sentence]] = determine_negverb(item[1])
                idx_to_negverb[sentence_to_idx[right_sentence]] = determine_negverb(item[2])

                idx_to_negnoun2[sentence_to_idx[left_sentence]] = determine_negnoun2(item[1])
                idx_to_negnoun2[sentence_to_idx[right_sentence]] = determine_negnoun2(item[2])

                idx_to_verbquant2[sentence_to_idx[left_sentence]] = determine_verbquant2(item[1])
                idx_to_verbquant2[sentence_to_idx[right_sentence]] = determine_verbquant2(item[2])

                idx_to_length[sentence_to_idx[left_sentence]] = len(item[1])
                idx_to_length[sentence_to_idx[right_sentence]] = len(item[2])


            elif model_type == 'trntn':
                left_sentence = ' '.join(item[1].leaves())
                right_sentence = ' '.join(item[2].leaves())

                idx_to_sentence[sentence_to_idx[left_sentence]] = left_sentence
                idx_to_sentence[sentence_to_idx[right_sentence]] = right_sentence

                idx_to_vector[sentence_to_idx[left_sentence]] = trntn.get_sentence_vector(item[1])
                idx_to_vector[sentence_to_idx[right_sentence]] = trntn.get_sentence_vector(item[2])

                idx_to_verb[sentence_to_idx[left_sentence]] = determine_verb(item[1].leaves())
                idx_to_verb[sentence_to_idx[right_sentence]] = determine_verb(item[2].leaves())

                idx_to_quant1[sentence_to_idx[left_sentence]] = determine_first_quantifier(item[1].leaves())
                idx_to_quant1[sentence_to_idx[right_sentence]] = determine_first_quantifier(item[2].leaves())

                idx_to_quant2[sentence_to_idx[left_sentence]] = determine_second_quantifier(item[1].leaves())
                idx_to_quant2[sentence_to_idx[right_sentence]] = determine_second_quantifier(item[2].leaves())

                idx_to_bothquants[sentence_to_idx[left_sentence]] = determine_bothquants(item[1].leaves())
                idx_to_bothquants[sentence_to_idx[right_sentence]] = determine_bothquants(item[2].leaves())

                idx_to_noun1[sentence_to_idx[left_sentence]] = determine_first_noun(item[1].leaves())
                idx_to_noun1[sentence_to_idx[right_sentence]] = determine_first_noun(item[2].leaves())

                idx_to_noun2[sentence_to_idx[left_sentence]] = determine_second_noun(item[1].leaves())
                idx_to_noun2[sentence_to_idx[right_sentence]] = determine_second_noun(item[2].leaves())

                idx_to_np2[sentence_to_idx[left_sentence]] = determine_last_np(item[1].leaves())
                idx_to_np2[sentence_to_idx[right_sentence]] = determine_last_np(item[2].leaves())

                idx_to_negquant1[sentence_to_idx[left_sentence]] = determine_negquant1(item[1].leaves())
                idx_to_negquant1[sentence_to_idx[right_sentence]] = determine_negquant1(item[2].leaves())

                idx_to_negnoun1[sentence_to_idx[left_sentence]] = determine_negnoun1(item[1].leaves())
                idx_to_negnoun1[sentence_to_idx[right_sentence]] = determine_negnoun1(item[2].leaves())

                idx_to_negverb[sentence_to_idx[left_sentence]] = determine_negverb(item[1].leaves())
                idx_to_negverb[sentence_to_idx[right_sentence]] = determine_negverb(item[2].leaves())

                idx_to_negnoun2[sentence_to_idx[left_sentence]] = determine_negnoun2(item[1].leaves())
                idx_to_negnoun2[sentence_to_idx[right_sentence]] = determine_negnoun2(item[2].leaves())

                idx_to_verbquant2[sentence_to_idx[left_sentence]] = determine_verbquant2(item[1].leaves())
                idx_to_verbquant2[sentence_to_idx[right_sentence]] = determine_verbquant2(item[2].leaves())

                idx_to_fullverbquant2[sentence_to_idx[left_sentence]] = determine_fullverbquant2(item[1].leaves())
                idx_to_fullverbquant2[sentence_to_idx[right_sentence]] = determine_fullverbquant2(item[2].leaves())

                idx_to_length[sentence_to_idx[left_sentence]] = len(item[1].leaves())
                idx_to_length[sentence_to_idx[right_sentence]] = len(item[2].leaves())

            rel_count += 1

print('Count: ', str(rel_count))
vectors_to_plot = torch.stack(idx_to_vector.values()).data.numpy()

def pca(vectors, out_dim):
    pca = PCA(n_components=out_dim)
    pca.fit(vectors)
    projections = pca.transform(vectors)
    idx_to_pca_vector = {idx: projections[idx, :] for idx in range(vectors.shape[0])}
    return (idx_to_pca_vector)

def tsne(vectors, out_dim):
    tsne = TSNE(n_components=out_dim)
    projections = tsne.fit_transform(vectors)
    idx_to_tsne_vector = {idx: projections[idx, :] for idx in range(vectors.shape[0])}
    return (idx_to_tsne_vector)

idx_to_projection = pca(vectors_to_plot, 2)

red5 = '#67001f'
red4 = '#b2182b'
red3 = '#d6604d'
red2 = '#f4a582'
red1 = '#fddbc7'
neutral = '#f7f7f7'
blue1 = '#d1e5f0'
blue2 = '#92c5de'
blue3 = '#4393c3'
blue4 = '#2166ac'
blue5 = '#053061'

colors = [red3, blue3, red1, blue1]
twocolors = [red3,blue3]
eightcolors = [red5, red4, red3, red1, blue5, blue4, blue3, blue1]
fivecolors = [red5, red3, blue5, blue3, blue1]
tencolors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

verbs = ['fear', 'hate', 'like', 'love']
quants = ['some', 'all', 'not_some', 'not_all']
bothquants = ['somesome', 'allsome', 'not_somesome', 'not_allsome', 'someall', 'allall', 'not_someall', 'not_allall']
nouns = ['Romans', 'Italians', 'Germans', 'Europeans', 'children']
nps = ['Romans', 'Italians', 'Germans', 'Europeans', 'children','notRomans', 'notItalians', 'notGermans', 'notEuropeans', 'notchildren']
adverbs = ['not', '']
verbquant2s = ['fear/hateall', 'like/loveall', 'fear/hatesome', 'like/lovesome']

verb_to_color = {'fear':red2, 'hate':red3,'like': blue2, 'love':blue3}
quant_to_color = {quant : colors[quants.index(quant)] for quant in quants}
bothquant_to_color = {bothquant : eightcolors[bothquants.index(bothquant)] for bothquant in bothquants}
noun_to_color = {noun : fivecolors[nouns.index(noun)] for noun in nouns}
np_to_color = {np : tencolors[nps.index(np)] for np in nps}
neg_to_color = {adv : twocolors[adverbs.index(adv)] for adv in adverbs}
quant2_to_color = {'some':twocolors[0], 'all':twocolors[1]}
length_to_color = {5: blue1, 6: blue2, 7: blue3, 8: blue4, 9: blue5}
verbquant2_to_color = {'fear/hateall':red3, 'like/loveall':red2,'fear/hatesome': blue3, 'like/lovesome':blue2}
fullverbquant2_to_color = {'fearall':red4, 'hateall': red3,'likeall':red2, 'loveall': red1,'fearsome': blue4, 'hatesome': blue3, 'likesome':blue2, 'lovesome': blue1}

def plot(n):
    if n == 2:
        for point in idx_to_projection.items():
            idx = point[0]
            #plt.scatter(point[1][0], point[1][1], color=noun_to_color[idx_to_noun1[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=noun_to_color[idx_to_noun2[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=verb_to_color[idx_to_verb[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=quant_to_color[idx_to_quant1[idx]], s=2)
            plt.scatter(point[1][0], point[1][1], color=quant2_to_color[idx_to_quant2[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=np_to_color[idx_to_np2[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=neg_to_color[idx_to_negquant1[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=neg_to_color[idx_to_negnoun1[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=neg_to_color[idx_to_negverb[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=neg_to_color[idx_to_negnoun2[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=length_to_color[idx_to_length[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=bothquant_to_color[idx_to_bothquants[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=verbquant2_to_color[idx_to_verbquant2[idx]], s=2)
            #plt.scatter(point[1][0], point[1][1], color=fullverbquant2_to_color[idx_to_fullverbquant2[idx]], s=2)

            # to annotate points:
            # plt.annotate(idx_to_sentence[idx], (point[1][0], point[1][1]), size=3)

    elif n == 3:
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        for point in idx_to_projection.items():
            #z = projections[:, 2]

            # reorder the labels to have colors matching the cluster results
            #y = np.choose(y, [1, 2, 0]).astype(np.float)
            ax.scatter(point[1][0], point[1][1], point[1][2])

    ymin, ymax = plt.gca().get_ylim()
    xmin, xmax = plt.gca().get_xlim()

    legend_x, legend_y = 2 * xmax, 2 * ymax

    # for verb
    # line_1 = plt.scatter(legend_x, legend_y, label='Line 1', color=red2, s=15)
    # line_2 = plt.scatter(legend_x, legend_y, label='Line 2', color=red3, s=15)
    # line_3 = plt.scatter(legend_x, legend_y, label='Line 3', color=blue2, s=15)
    # line_4 = plt.scatter(legend_x, legend_y, label='Line 4', color=blue3, s=15)
    # plt.legend([line_1, line_2, line_3, line_4], verbs)  # numpoints=1)

    # for quant2
    line_1 = plt.scatter(legend_x, legend_y, label='Line 1', color=twocolors[0], s=15)
    line_2 = plt.scatter(legend_x, legend_y, label='Line 2', color=twocolors[1], s=15)
    plt.legend([line_1, line_2], ['some', 'all'])

    # for verbquant2
    # line_1 = plt.scatter(legend_x, legend_y, label='Line 1', color=red3, s=15)
    # line_2 = plt.scatter(legend_x, legend_y, label='Line 2', color=red2, s=15)
    # line_3 = plt.scatter(legend_x, legend_y, label='Line 3', color=blue3, s=15)
    # line_4 = plt.scatter(legend_x, legend_y, label='Line 4', color=blue2, s=15)
    # plt.legend([line_1, line_2, line_3, line_4], ['fear/hate all', 'like/love all', 'fear/hate some', 'like/love some'])

    # for full verbquant2
    # line_1 = plt.scatter(legend_x, legend_y, label='Line 1', color=red4, s=15)
    # line_2 = plt.scatter(legend_x, legend_y, label='Line 2', color=red3, s=15)
    # line_3 = plt.scatter(legend_x, legend_y, label='Line 3', color=red2, s=15)
    # line_4 = plt.scatter(legend_x, legend_y, label='Line 4', color=red1, s=15)
    # line_5 = plt.scatter(legend_x, legend_y, label='Line 5', color=blue4, s=15)
    # line_6 = plt.scatter(legend_x, legend_y, label='Line 6', color=blue3, s=15)
    # line_7 = plt.scatter(legend_x, legend_y, label='Line 7', color=blue2, s=15)
    # line_8 = plt.scatter(legend_x, legend_y, label='Line 8', color=blue1, s=15)
    # plt.legend([line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8], ['fear all', 'hate all','like all','love all','fear some','hate some', 'like some', 'love some'])

    plt.xlim(1.05 * xmin, 1.05 * xmax)
    plt.ylim(1.05 * ymin, 1.05 * ymax)

    #plt.title('Verb')
    plt.savefig('sentencevectors-trntn-binaryfol-quant2b.png', format='png', dpi=500)
    plt.show()