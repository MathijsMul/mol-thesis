"""
load and test pretrained model
"""

import torch
from rnn import RNN
from sumnn import sumNN
from trnn import tRNN
from trntn import tRNTN
import datamanager as dat
from test import compute_accuracy
#from visualize import confusion_matrix

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
#vocab += ['Venetians', 'Belgians', 'women', 'Dutch', 'Polish', 'Milanese', 'Neapolitans', 'Spanish', 'Parisians', 'Russians', 'students', 'linguists']

rels = ['#', '<', '=', '>', '^', 'v', '|']

word_dim = 50
n_hidden = 128
cpr_dim = 75

test_files_glove = [
    '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairsreligion.txt'
]

test_files = ['/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/train578_test69/binary_2dets_4negs_train578_test69_test_0bracket_pairs.txt']

# best glove model
model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/GRUbinary_2dets_4negs_train_0bracket_pairs1.pt'
nn = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim, use_glove=True)
nn.load_state_dict(torch.load(model_path))

for test_data_file in test_files_glove:
    test_data = dat.SentencePairsDataset(test_data_file)
    test_data.load_data(sequential=True)
    test_acc = compute_accuracy(test_data,rels, nn, print_outputs=False, confusion_matrix=False, length_matrix=False, batch_size=200)
    print(test_acc)