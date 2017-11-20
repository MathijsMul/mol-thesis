import torch
from trnn import tRNN
from trntn import tRNTN
from sumnn import sumNN
import datamanager as dat
from test import compute_accuracy, comp_acc_per_length
#from visualize import confusion_matrix

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
#vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some', 'two']
rels = ['#', '<', '=', '>', '^', 'v', '|']
#rels = ['#', '<', '=', '>', '|']
word_dim = 25
cpr_dim = 75
bound_layers = None
bound_embeddings = None

# to load model:
# sumnn_path = 'models/sumnn/sumNNbinary_2dets_4negs_578_train.pt'
# sumnn = sumNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
# sumnn.load_state_dict(torch.load(sumnn_path))

trnn_path = 'models/trnn/tRNNbinary_2dets_4negs_578_train.pt'
trnn = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
trnn.load_state_dict(torch.load(trnn_path))

trntn_path = 'models/trntn/tRNTNbinary_2dets_4negs_578_train.pt'
trntn = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
trntn.load_state_dict(torch.load(trntn_path))

#test_data_file = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
test_data_file = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/2dets_4negs/578/binary_2dets_4negs_56789_test.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data()

final_acc = comp_acc_per_length(test_data, rels, trnn, 5, 9, threed_plot=True)
print(final_acc)

# for net in [sumnn,trnn,trntn]:
#     final_acc = compute_accuracy(test_data, rels, net, print_outputs=False, confusion_matrix=True)
#     print(final_acc)

