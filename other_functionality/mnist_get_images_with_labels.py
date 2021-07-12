import torch
import torchvision
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


batch_size = 100

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=ToTensor())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=ToTensor())
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0)

dataiter = iter(trainloader)
images, labels = dataiter.next()

my_dict = {}

for image, label in zip(images, labels):

    if label not in my_dict.keys():
        my_dict[str(label)] = image
    else:
        continue

print(my_dict.keys())

fig, axs = plt.subplots(2, 5, figsize=(20, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.001)

axs = axs.ravel()

for i in range(10):
    img = my_dict.get('tensor('+str(i)+')') / 2 + 0.5
    npimg = img.numpy().reshape(28, 28)

    axs[i].imshow(npimg, cmap='gray')
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    axs[i].set_title("Digit: "+str(i))

fig.suptitle("Examples of MNIST Digits", fontsize=14)
plt.show()
