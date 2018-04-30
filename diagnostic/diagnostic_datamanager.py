from __future__ import print_function, division
import torch
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
        # shuffle again
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

        return (train_dataset, test_dataset)