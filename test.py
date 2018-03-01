from torch import nn
import torch
from torch.autograd import Variable
from torchvision import datasets,transforms
import torch.optim as optim

batch_size = 50
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

#data,target = iter(train_loader).next()
#o = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
#o = nn.Linear(10,10)
#data, target = Variable(data), Variable(target)

#print(o)
#print(o(x))
num_layers = 4
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.first_layer = nn.Linear(28*28,10)
        self.mainlayers = nn.ModuleList([nn.Linear(10, 10) for i in range(num_layers)])


    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = torch.nn.functional.relu(self.first_layer(x))
        for i in (range(num_layers)):
            x = torch.nn.functional.relu(self.mainlayers[i](x))
        x = torch.nn.functional.log_softmax(x)
        return x

m = MnistModel()
m.train()
train_loss = []
train_accu = []
i = 0
optimizer = optim.Adam(m.parameters(), lr=0.01)
for epoch in range(2):
    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = m(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        prediction = output.data.max(1)[1]
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        train_accu.append(accuracy)
        if (i % 1000 == 0):
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data[0], accuracy))
        i += 1
        #print(i)
#print(output)
#print(accuracy)