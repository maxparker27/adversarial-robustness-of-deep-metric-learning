import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from pytorch_metric_learning import losses, reducers, miners, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from dml_model_class_mnist import CNN
import foolbox as fb
from foolbox import samples, accuracy
import numpy as np
from dml_adversarial_attacks import determine_attack, perform_attack
from deep_metric_learning_functions import test, train


print("Successfully imported libraries")

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

print(training_data)

# Download test data from open datasets.
test_data = datasets.MNIST(
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

# Get cpu or gpu device for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

in_channel = X.shape[1]
num_classes = 10

# Define model


model = CNN().to(device)
# ------------------------------------------------------------------------------------------------
# Deep Metric Learning - Triplet Margin Loss:
# distance = distances.CosineSimilarity()

# reducer = reducers.ThresholdReducer(low=0)

# loss_fn = losses.TripletMarginLoss(
#     margin=0.2, distance=distance, reducer=reducer)

# miner = miners.TripletMarginMiner(
#     margin=0.2, distance=distance, type_of_triplets="semihard")

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

# num_epochs = 20
# ------------------------------------------------------------------------------------------------
# Deep Metric Learning - Contrastive Loss:
distance = distances.CosineSimilarity()

reducer = reducers.ThresholdReducer(low=0)

loss_fn = losses.ContrastiveLoss(
    pos_margin=1, neg_margin=0, distance=distance, reducer=reducer)

miner = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

num_epochs = 1

print(model)
# ------------------------------------------------------------------------------------------------


for epoch in range(1, num_epochs+1):
    train(train_dataloader, model, loss_fn, optimizer, miner, device)
    test(training_data, test_data, model, accuracy_calculator)

PATH = './mnist_dml_model.pth'
torch.save(model.state_dict(), PATH)

model.eval()

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
