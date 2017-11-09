import torch
from trnn import tRNN
from trntn import tRNTN
import datamanager as dat
from test import compute_accuracy
#from visualize import confusion_matrix

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']
#rels = ['#', '<', '=', '>', '|']
word_dim = 25
cpr_dim = 75
bound_layers = None
bound_embeddings = None

# to load model:
net = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
net.load_state_dict(torch.load('models/trntn/tRNTNbinary1_neg_verb_train.pt'))

test_data_file = 'data/binary/negate_noun1/split/binary1_neg_noun1_test.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data()

final_acc = compute_accuracy(test_data, rels, net, print_outputs=False, confusion_matrix=True)
print(final_acc)

