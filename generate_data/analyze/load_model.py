import torch
from rnn import RNN
from sumnn import sumNN
from trnn import tRNN
from trntn import tRNTN
import datamanager as dat
from test import compute_accuracy
#from visualize import confusion_matrix

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']

word_dim = 25
n_hidden = 128
cpr_dim = 75

#test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data(sequential=True)

#model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol/models/tRNTNbinary_2dets_4negs_train4.pt'
model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs1.pt'
nn = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
nn.load_state_dict(torch.load(model_path))
compute_accuracy(test_data,rels, nn, print_outputs=False, confusion_matrix=True, batch_size=200)