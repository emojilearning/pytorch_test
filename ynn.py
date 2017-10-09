import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.optim as optim
import ministdataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        # self.fc4   = nn.Linear(28*28,200)
        # self.fc5   = nn.Linear(200,10)

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.relu(self.fc4(x))
        # x = self.fc5(x)
        return x

def test(model):
    total = 0
    correct = 0
    ynn_instance = model
    for data in ministdataset.testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = ynn_instance(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

def main():
    net = Net().cuda()
    #nn.MSELoss
    criterion = nn.CrossEntropyLoss()
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr = 0.01, weight_decay=0)
    print("begin")
    # in your training loop:
    for epoch in range(0,11):
        running_loss = 0    
        # print("first")
        for i,data in enumerate(ministdataset.trainloader,0):
            inputs, labels = data
            # print("run a iter")
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # print(inputs)
            optimizer.zero_grad() # zero the gradient buffers
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step() # Does the update
            # print statistics
            running_loss += loss.data[0]
            
            if i % 2000 == 1999: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0
        test(net)

    torch.save(net, 'model1.pkl')



    #model = torch.load('model.pkl').state_dict(), 'params.pkl'

if __name__ == '__main__':
    main()
    

