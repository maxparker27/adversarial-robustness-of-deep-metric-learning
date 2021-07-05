import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import foolbox as fb
import eagerpy as ep
from foolbox import samples, accuracy
import numpy as np


print("Successfully imported libraries")

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


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


class CNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(
            3, 3), padding=(1, 1), stride=(1, 1))
        self.batchNormalization1 = nn.BatchNorm2d(num_features=8)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(
            3, 3), padding=(1, 1), stride=(1, 1))
        self.batchNormalization2 = nn.BatchNorm2d(num_features=16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
            5, 5), padding=(1, 1), stride=(2, 2))
        self.batchNormalization3 = nn.BatchNorm2d(num_features=32)

        self.dropout1 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
            3, 3), padding=(1, 1), stride=(1, 1))
        self.batchNormalization4 = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(
            3, 3), padding=(1, 1), stride=(1, 1))
        self.batchNormalization5 = nn.BatchNorm2d(num_features=128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(
            5, 5), padding=(1, 1), stride=(2, 2))
        self.batchNormalization6 = nn.BatchNorm2d(num_features=256)

        self.dropout2 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(9216, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchNormalization1(x)
        x = F.relu(self.conv2(x))
        x = self.batchNormalization2(x)
        x = F.relu(self.conv3(x))
        x = self.batchNormalization3(x)

        x = self.dropout1(x)

        x = F.relu(self.conv4(x))
        x = self.batchNormalization4(x)
        x = F.relu(self.conv5(x))
        x = self.batchNormalization5(x)
        x = F.relu(self.conv6(x))
        x = self.batchNormalization6(x)

        x = self.dropout2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


model = CNN().to(device)

# ------------------------------------------------------------------------------------------------
# Standard Classification Loss:
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(X)
        loss = loss_fn(outputs, y)

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

PATH = './mnist_dl_model.pth'
torch.save(model.state_dict(), PATH)

model = CNN()
model.load_state_dict(torch.load(PATH))
model.eval()


images, labels = [], []

for image, label in test_data:

    images.append(image.tolist())
    labels.append(label)

images = torch.Tensor(images).to(device)
labels = torch.Tensor(labels).type(torch.LongTensor).to(device)

print(images.shape)
print(labels.shape)


def determine_attack(which_attack):
    if which_attack == "FGSM":
        return fb.attacks.LinfFastGradientAttack()
    elif which_attack == "CarliniWagner":
        return fb.attacks.L2CarliniWagnerAttack()
    elif which_attack == "PGD":
        return fb.attacks.LinfProjectedGradientDescentAttack()


def perform_attack(choose_attack, model, epsilons):

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = determine_attack(choose_attack)
    print("\nUsing {} attack:".format(choose_attack))

    for epsilon in epsilons:
        raw_advs, clipped_advs, success = attack(
            fmodel, images, labels, epsilons=epsilon)

        success = success.type(torch.FloatTensor)

        clean_acc = accuracy(fmodel, images, labels)
        print(f"clean accuracy:  {clean_acc * 100:.1f} %")

        robust_accuracy = 1 - success.mean(axis=-1)
        print("robust accuracy for perturbations with epsilon = {} is {}%.".format(
            epsilon, round(robust_accuracy.item() * 100, 4)))
    print("{} attack is complete.".format(choose_attack))


perform_attack(choose_attack="FGSM", model=model, epsilons=[
               0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

# need to update which default parameters of CW to use
# perform_attack(choose_attack = "CarliniWagner", model = model, epsilons = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
perform_attack(choose_attack="PGD", model=model, epsilons=[
               0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])


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


test_adversarials(raw_advs, labels, model)
