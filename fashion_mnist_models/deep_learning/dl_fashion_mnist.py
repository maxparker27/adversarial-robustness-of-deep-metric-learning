from deep_learning_functions import train, test
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import foolbox as fb
from foolbox import samples, accuracy
import numpy as np
from model_class_fashion_mnist import FashionMNISTCNN
from adversarial_attacks import perform_attack, determine_attack

print("Successfully imported libraries")

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 256
learning_rate = 0.01

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Choose which device to use for training - either CPU or GPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

num_classes = 10  # Number of outputs from the model

# Define model - The model file is being imported from another file in the directory
model = FashionMNISTCNN().to(device)

# ------------------------------------------------------------------------------------------------
# Standard Classification Loss:
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model)


epochs = 1  # Setting number of epochs for model to train

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # Training function is defined in deep_learning_functions.py
    train(train_dataloader, model, loss_fn, optimizer, device)

    # Test function is defined in deep_learning_functions.py
    test(test_dataloader, model, loss_fn, device)
print("Done!")

PATH = './mnist_dl_model.pth'
torch.save(model.state_dict(), PATH)

# model = CNN()
# model.load_state_dict(torch.load(PATH))
# model.eval()

images, labels = [], []

for image, label in test_data:

    # Appending images to images list in order to use later for attacks
    images.append(image.tolist())

    # Appending labels to labels list in order to use later for attacks
    labels.append(label)


images = torch.Tensor(images).to(device)  # Sending images to either GPU or CPU
labels = torch.Tensor(labels).type(torch.LongTensor).to(
    device)  # Sending labels to either GPU or CPU

print("\nThe shape of the images list is: {}".format(images.shape))
print("The shape of the labels list is: {}\n".format(labels.shape))


perform_attack("FGSM", model, epsilons=[
               0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], images=images, labels=labels)

# need to update which default parameters of CW to use
# perform_attack(choose_attack = "CarliniWagner", model = model, epsilons = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
perform_attack("PGD", model, epsilons=[
               0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], images=images, labels=labels)


# fix this function -> seems to be giving a weird output
def test_adversarials(adversarials, labels, model):
    size = len(test_dataloader.dataset)
    model.eval()
    correct = 0
    for X, y in zip(adversarials, labels):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        print(pred)
        break
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%")


# test_adversarials(raw_advs, labels, model)
