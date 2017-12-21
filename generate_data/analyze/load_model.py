import torch
from rnn import RNN
import datamanager as dat
from test import compute_accuracy
#from visualize import confusion_matrix

vocab =   ['(', ')', 'Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']

word_dim = 25
n_hidden = 128
cpr_dim = 75
#
# print('100% brackets')

#test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_train.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data(sequential=True)
#
# srn = RNN('SRN', vocab, rels, word_dim, n_hidden, cpr_dim)
# srn.load_state_dict(torch.load('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/srn/SRNbinary_2dets_4negs_train_ada_nodrop_3.pt'))
# final_acc = compute_accuracy(test_data,rels, srn, print_outputs=False, confusion_matrix=False)
# print(final_acc)
#
model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/gru/GRUbinary_2dets_4negs_train_ada_nodrop_3.pt'
gru = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
gru.load_state_dict(torch.load(model_path))
final_acc = compute_accuracy(test_data,rels, gru, print_outputs=False, confusion_matrix=False, batch_size=200)
print(final_acc)
#
# lstm = RNN('LSTM', vocab, rels, word_dim, n_hidden, cpr_dim)
# lstm.load_state_dict(torch.load('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/LSTM/LSTMbinary_2dets_4negs_trainadad_dropout1.pt'))
#
# final_acc = compute_accuracy(test_data,rels, lstm, print_outputs=False, confusion_matrix=False)
# print(final_acc)

# print('50% brackets')
# test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0.5bracket_pairs.txt'
# test_data = dat.SentencePairsDataset(test_data_file)
# test_data.load_data(sequential=True)
#
# srn = RNN('SRN', vocab, rels, word_dim, n_hidden, cpr_dim)
# srn.load_state_dict(torch.load('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/srn/partial_bracketting/SRNbinary_2dets_4negs_train_0.5bracketpairs1nodrop.pt'))
# final_acc = compute_accuracy(test_data,rels, srn, print_outputs=False, confusion_matrix=False)
# print(final_acc)
#
# gru = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
# gru.load_state_dict(torch.load('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/gru/partial_bracketting/GRUbinary_2dets_4negs_train_0.5bracketpairs1nodrop.pt'))
# final_acc = compute_accuracy(test_data,rels, gru, print_outputs=False, confusion_matrix=False)
# print(final_acc)
#
# lstm = RNN('LSTM', vocab, rels, word_dim, n_hidden, cpr_dim)
# lstm.load_state_dict(torch.load('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/lstm/partial_bracketting/LSTMbinary_2dets_4negs_train_0.5bracketpairs1nodrop.pt'))
# final_acc = compute_accuracy(test_data,rels, lstm, print_outputs=False, confusion_matrix=False)
# print(final_acc)

#
# print('0% brackets')
# vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
# test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt'
# test_data = dat.SentencePairsDataset(test_data_file)
# test_data.load_data(sequential=True)
#
# srn = RNN('SRN', vocab, rels, word_dim, n_hidden, cpr_dim)
# srn.load_state_dict(torch.load('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/srn/partial_bracketting/SRNbinary_2dets_4negs_train_0bracket_pairs1nodrop.pt'))
# final_acc = compute_accuracy(test_data,rels, srn, print_outputs=False, confusion_matrix=False)
# print(final_acc)
#
# gru = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
# gru.load_state_dict(torch.load('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/gru/partial_bracketting/GRUbinary_2dets_4negs_train_0bracket_pairs1nodrop.pt'))
# final_acc = compute_accuracy(test_data,rels, gru, print_outputs=False, confusion_matrix=False)
# print(final_acc)
#
# lstm = RNN('LSTM', vocab, rels, word_dim, n_hidden, cpr_dim)
# lstm.load_state_dict(torch.load('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/lstm/partial_bracketting/LSTMbinary_2dets_4negs_train_0bracket_pairs1nodrop.pt'))
# final_acc = compute_accuracy(test_data,rels, lstm, print_outputs=False, confusion_matrix=False)
# print(final_acc)
#
