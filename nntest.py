import torch
import time
import ministdataset
from torch.autograd import Variable

from ynn import Net

total = 0
correct = 0
ynn_instance = torch.load('model1.pkl')
t = time.time()
for data in ministdataset.testloader:
    images, labels = data
    outputs = ynn_instance(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print(time.time()-t)
print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
