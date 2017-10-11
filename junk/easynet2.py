from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class Basic(nn.Module):

    def __init__(self):
        super().__init__()

        self.w = nn.Linear(5, 10)

        # matrix to softmax layer
        self.sm = nn.Linear(10, 2)

    def forward(self, inputs):
        #print(inputs)
        inputs = Variable(torch.FloatTensor(inputs).view(1,5))
        #print(inputs)
        output = self.w(inputs)
        output = self.sm(output)
        return(output)

net = Basic()
optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
#
targets = []
data = []

# targets = torch.LongTensor(1,500)
# data = torch.FloatTensor(500,5)

with open('easydata.txt', 'r') as f:
    for idx, line in enumerate(f):
        all = line.split('\t')
        targets.append([int(all[0])])
        data.append([float(all[i]) for i in range(1,6)])

        # label = torch.LongTensor([int(all[0])])
        # dat = torch.FloatTensor([float(all[i]) for i in range(1, 6)])
        # #dat = torch.Tensor(dat)
        #label = torch.LongTensor([label])

        # targets[:,idx] = label
        # data[idx,:] = dat
        # targets.view(500)

print(targets)
print(data)

#print(torch.Tensor([0.9585240680113926, 0.5232392115643326, 0.9523791656998428, 0.38281500613077457, 0.13965298549625693]))

for epoch in range(100):
    for i in range(500):
 #       inputs, target = data[i,:], targets[:,i]
        inputs, target = data[i], targets[i]

        #inputs, target = Variable(inputs), Variable(target)

        target = Variable(torch.LongTensor(target))

        optimizer.zero_grad()

        outputs = net(inputs)
        outputs = F.softmax(outputs.view(1,2))

        # print('outputs')
        # print(outputs)
        #
        # print('target')
        # print(target)

        loss = criterion(outputs, target)
        #
        # print('loss')
        #print(loss)
        print(outputs)
        loss.backward()
        optimizer.step()
        #print(list(net.parameters()))

# Mathijs:
# this all workings fine, proving that forward DOES NOT require tensors, but can also accept other
# (undifferentiable) data structures
