import torchvision
import torchvision.transforms as transforms
import torch
# from matplotlib import pyplot as plt
import numpy as np

# torchvision数据集的输出是在[0, 1]范围内的PILImage图片。
# 我们此处使用归一化的方法将其转化为Tensor，数据范围为[-1, 1]

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                             ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, 
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=40, 
                                          shuffle=False)
classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')



# def imshow(img):
#     plt.figure('model')
#     img = img / 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#     plt.show()
    
# def main():
# # functions to show an image
#     import matplotlib.pyplot as plt
#     import numpy as np
#     # show some random training images
#     dataiter = iter(trainloader)
#     images, labels = dataiter.next()
#     # print images
#     imshow(torchvision.utils.make_grid(images))

# if __name__ == '__main__':
#     main()