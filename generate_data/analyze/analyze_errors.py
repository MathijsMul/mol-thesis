import torch
from trnn import tRNN
from trntn import tRNTN
from sumnn import sumNN
import datamanager as dat
from test import comp_error_matrix

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']
word_dim = 25
cpr_dim = 75
bound_layers = None
bound_embeddings = None

# to load model:
sumnn_path1 = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/sumnn/sumNNbinary_2dets_4negs_train.pt'
sumnn = sumNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
sumnn.load_state_dict(torch.load(sumnn_path1))

trnn_path1 = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/trnn/tRNNbinary_2dets_4negs_train.pt'
trnn = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
trnn.load_state_dict(torch.load(trnn_path1))

trntn_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/trntn/tRNTNbinary_2dets_4negs_train.pt'
trntn = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
trntn.load_state_dict(torch.load(trntn_path))

test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
#test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/junk/minitest_bin.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data()

comp_error_matrix(test_data, rels, sumnn, show_totals=True)

comp_error_matrix(test_data, rels, trnn, show_totals=True)

comp_error_matrix(test_data, rels, trntn, show_totals=True)
#comp_error_matrix(test_data, rels, trnn)

