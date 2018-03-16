import torch
from rnn import RNN
from trntn import tRNTN
import datamanager as dat
from test import compute_accuracy

vocab = ['(', ')', 'Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']

word_dim = 25
n_hidden = 128
cpr_dim = 75

test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data(sequential=True)

model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/gru/GRUbinary_2dets_4negs_train_ada_nodrop_3.pt'
gru = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
untrained_dict = gru.state_dict()

# make random GRU: trained comparison + softmax layer, untrained recurrent unit
trained_dict = torch.load(model_path)
rnn_layers = ['rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'rnn.bias_ih_l0', 'rnn.bias_hh_l0']
trained_dict_without_rnn = {k: v for k, v in trained_dict.items() if k not in rnn_layers}

untrained_dict.update(trained_dict_without_rnn)
gru.load_state_dict(untrained_dict)

final_acc_random_rnn = compute_accuracy(test_data,rels, gru, print_outputs=False, confusion_matrix=False, batch_size=200)
print(final_acc_random_rnn)

gru.load_state_dict(trained_dict)
final_acc_trained_rnn = compute_accuracy(test_data,rels, gru, print_outputs=False, confusion_matrix=False, batch_size=200)
print(final_acc_trained_rnn)
