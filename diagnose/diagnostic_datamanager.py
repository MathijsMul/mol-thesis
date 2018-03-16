from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import ast
from random import shuffle
import math

class DiagnosticDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, item):
        return(self.data[item])

class DiagnosticDataFile(Dataset):

    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        print('Loading data from ' + self.data_file)
        set_all_data = set()
        list_all_data = []
        total = 0
        with open(self.data_file, 'r') as file:
            for idx, line in enumerate(file):
                # no duplicates
                set_all_data.add(line)
                #list_all_data += [line]

                total += 1

        print('Total nr. instances: ', str(total))

        all_data = []
        for line in set_all_data:
        #for line in list_all_data:
            [label, data] = line.split('\t')
            label = int(label)
            data = ast.literal_eval(data)
            tensor_data = torch.Tensor(data)
            tensor_label = torch.LongTensor([label])

            all_data += [[tensor_label, tensor_data]]
        return(all_data)

    def shuffle(self):
        shuffle(self.data)

    def split(self, train_ratio):
        #TODO: shuffle again
        self.shuffle()
        split_idx = math.ceil(train_ratio * len(self))
        train_dataset = DiagnosticDataset(self.data[:split_idx])
        test_dataset = DiagnosticDataset(self.data[split_idx:])
        return(train_dataset, test_dataset)

    def alternative_split(self, train_ratio):
        self.shuffle()
        split_idx = math.ceil(train_ratio * len(self))
        train_dataset = DiagnosticDataset(self.data[:split_idx])
        test_dataset = DiagnosticDataset(self.data[split_idx:])
        # take test instances out of training set:

        return (train_dataset, test_dataset)


#data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/diagnose/small.txt'
# data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/diagnose/diagnostic_gru_2dets4negs_brackets.txt_downsampled_0.05'
# d = DiagnosticDataFile(data_file)
# train_data, test_data = d.split(0.8)
# print(len(train_data))
# #print(train_data[0])
# print(len(test_data))
#
# train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)
# for idx, item in enumerate(train_dataloader):
#     print([i.size() for i in item])
