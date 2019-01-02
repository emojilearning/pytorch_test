import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.optim as optim
import ministdataset
from time import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3,padding=1)
        self.pool  = nn.MaxPool2d(2,2)


        # self.conv2 = nn.Conv2d(16, 32, 3,padding=1)
        self.conv2 = nn.Sequential(nn.Conv2d(16,16,3,groups=16,padding=1),nn.Conv2d(16,32,1))
        # self.conv3 = nn.Conv2d(32, 64, 3,padding=1)
        self.conv3 = nn.Sequential(nn.Conv2d(32,32,3,groups=32,padding=1),nn.Conv2d(32,64,1))
        self.conv4 = nn.Conv2d(64, 10, 1,padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(-1, self.num_flat_features(x))
        
        return x

def test(model):
    t1 = time()
    model.eval()
    total = 0
    correct = 0
    ynn_instance = model
    for data in ministdataset.testloader:
        images, labels = data
        outputs = ynn_instance(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * float(correct) / total))
    print('elasped time: %f' % (time()-t1))
    model.train()

def main():
    cuda = torch.cuda.is_available()
    if cuda:
        model = Net().cuda()
    else:
        model = Net()

    model.train()
    # model.load_state_dict(torch.load('model1.pkl').state_dict(),False)
    #nn.MSELoss
    criterion = nn.CrossEntropyLoss()
    # create your optimizer
    optimizer = optim.SGD(model.parameters(), lr = 0.05, weight_decay=0.001,momentum=0)
    print("begin")
    # in your training loop:
    for epoch in range(0,11):
        running_loss = 0    
        # print("first")
        for i,data in enumerate(ministdataset.trainloader,0):
            inputs, labels = data
            # print("run a iter")
            inputs, labels = Variable(inputs), Variable(labels)
            # print(inputs)
            optimizer.zero_grad() # zero the gradient buffers
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step() # Does the update
            # print statistics
            running_loss += loss.data.item()
            
            if i % 2000 == 1999: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0
        test(model)
        torch.save(model,"model%d.pkl"%epoch)
    

    torch.save(model, 'model.pkl')



    #model = torch.load('model.pkl').state_dict(), 'params.pkl'

if __name__ == '__main__':
    main()
    

