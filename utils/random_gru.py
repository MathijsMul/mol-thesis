"""
construct and test random GRU (with untrained recurrent unit)
"""

import torch
from rnn import RNN
import datamanager as dat
from test import compute_accuracy

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']

word_dim = 25
n_hidden = 128
cpr_dim = 75

test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data(sequential=True)

model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs1.pt'
trained_dict = torch.load(model_path)
rnn_layers = ['rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'rnn.bias_ih_l0', 'rnn.bias_hh_l0']
trained_dict_without_rnn = {k: v for k, v in trained_dict.items() if k not in rnn_layers}

total_acc = 0
for i in range(5):
    # make random GRU: trained comparison + softmax layer, untrained recurrent unit
    gru = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
    untrained_dict = gru.state_dict()

    untrained_dict.update(trained_dict_without_rnn)
    gru.load_state_dict(untrained_dict)

    final_acc_random_rnn = compute_accuracy(test_data,rels, gru, print_outputs=False, confusion_matrix=False, batch_size=200)
    print(final_acc_random_rnn)
    total_acc += float(final_acc_random_rnn)

print('Avg testing acc random GRU:')
print(total_acc / 5)