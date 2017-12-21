import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
import math
from diagnostic_datamanager import DiagnosticDataset, DiagnosticDataFile

class DiagnosticClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_labels):
        super(DiagnosticClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        # but diagnostic classifier must be linear
        self.log_reg = nn.Linear(self.input_size, self.num_labels)
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.num_labels)
        self.tanh = nn.Tanh()
        #self.log_softmax = nn.LogSoftmax()
        self.log_softmax = nn.LogSoftmax()
        #self.linear = True

    def forward(self, inputs):

        input = Variable(inputs)
        to_softmax = self.log_reg(input)
        outputs = self.log_softmax(to_softmax)

        # not linear:
        # hidden = self.layer1(input)
        # hidden_act = self.tanh(hidden)
        # to_softmax = self.layer2(hidden_act)
        # outputs = self.log_softmax(to_softmax)
        return(outputs)  # size: batch_size x num_classes

#def train(model, num_epochs, data, optimizer, criterion, test_each_epoch):
def train(model, num_epochs, train_dataloader, optimizer, criterion, test_each_epoch, test_dataloader):
    model.train()

    for epoch in range(num_epochs):
        print('Training epoch ' + str(epoch + 1))

        for idx, item in enumerate(train_dataloader):

        # if epoch > 0 and shuffle_samples:
        #     data.batch_data(shuffle_samples)
        #
        # for batch in data.batched_data:

            #inputs = torch.stack([item[1] for item in batch])
            inputs = item[1]
            labels = Variable(item[0].squeeze(1))
            # print('labels')
            # print(labels)
            #labels = Variable(torch.squeeze(torch.stack([item[0] for item in batch]),1))
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print('outputs')
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if test_each_epoch:
            acc = compute_accuracy(test_dataloader,model)
            print(acc)

def compute_accuracy(test_dataloader, net):
    net.eval()

    correct = 0.0
    total = 0

    #for batch in test_data.batched_data:
    for idx, item in enumerate(test_dataloader):
        inputs = item[1]
        labels = item[0]
        outputs = net(inputs)

        # inputs = torch.stack([item[1] for item in batch])
        # labels = torch.squeeze(torch.stack([item[0] for item in batch]), 1)
        # outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        # print(labels)
        # print(predicted == labels.squeeze(1))
        correct += (predicted == labels.squeeze(1)).sum()
        #print(correct)
        total += len(inputs)

    acc = 100 * correct / total
    acc = "%.2f" % round(acc, 2)

    return (acc)

def train_regime_standard():

    #data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/diagnose/small.txt'
    data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/diagnose/diagnostic_gru_2dets4negs_brackets.txt_downsampled_0.4'
    d = DiagnosticDataFile(data_file)

    train_data, test_data = d.split(0.8)
    print(str(len(train_data)) + ' train instances')
    print(str(len(test_data)) + ' test instances')

    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=4)
    #print(set([item[0][0] for item in train_data.data]))
    num_classes = len(set([item[0][0] for item in train_data.data]))

    input_size = 128
    hidden_size = 64
    model = DiagnosticClassifier(input_size, hidden_size, num_classes)
    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adadelta(model.parameters())
    criterion = nn.NLLLoss()
    test_each_epoch = True
    num_epochs = 50
    train(model, num_epochs, train_dataloader, optimizer, criterion, test_each_epoch, test_dataloader)
    torch.save(model.state_dict(), 'first_diag_class2.pt')

train_regime_standard()

# DC = DiagnosticClassifier(128, 5)
# model_path = 'first_diag_class.pt'
# DC.load_state_dict(torch.load(model_path))
# #train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/diagnose/small.txt'
# train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/diagnose/diagnostic_gru_2dets4negs_brackets.txt_downsampled_0.05'
# data = DiagnosticDataset(train_data_file, 25)
# data.batch_data(False)
# acc = compute_accuracy(data,DC)
# print(acc)
#
