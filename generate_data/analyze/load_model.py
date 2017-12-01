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
# sumnn_path1 = 'models/sumnn/sumNNbinary1_neg_det1_train.pt'
# sumnn1 = sumNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
# sumnn1.load_state_dict(torch.load(sumnn_path1))

# sumnn_path2 = 'models/sumnn/sumNNbinary1_neg_noun1_train.pt'
# sumnn2 = sumNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
# sumnn2.load_state_dict(torch.load(sumnn_path2))

# sumnn_path3 = 'models/sumnn/sumNNbinary1_neg_verb_train.pt'
# sumnn3 = sumNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
# sumnn3.load_state_dict(torch.load(sumnn_path3))
#
# trnn_path1 = 'models/trnn/tRNNbinary1_neg_det1_train.pt'
# trnn1 = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
# trnn1.load_state_dict(torch.load(trnn_path1))
#
# trnn_path2 = 'models/trnn/tRNNbinary1_neg_verbtrain.pt'
# trnn2 = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
# trnn2.load_state_dict(torch.load(trnn_path2))


# trntn_path = 'models/trntn/tRNTNbinary_2dets_4negs_578_train.pt'
# trntn = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
# trntn.load_state_dict(torch.load(trntn_path))


trntn_path1 = 'models/trntn/tRNTNbinary1_neg_det1_train.pt'
trntn1 = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
trntn1.load_state_dict(torch.load(trntn_path1))

trntn_path2 = 'models/trntn/tRNTNbinary1_neg_verb_train.pt'
trntn2 = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
trntn2.load_state_dict(torch.load(trntn_path2))

#test_data_file = 'data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
test_data_file = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/2dets_4negs/578/binary_2dets_4negs_56789_test.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data()

# final_acc = comp_acc_per_length(test_data, rels, trnn, 5, 9, threed_plot=True)
# print(final_acc)

for net in [trntn1, trntn2]:
    final_acc = compute_accuracy(test_data, rels, net, print_outputs=False, confusion_matrix=False)
    print(final_acc)

