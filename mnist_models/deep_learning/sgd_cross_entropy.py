# from deep_learning_functions import train, test
import pandas as pd
import numpy as np
import foolbox as fb
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch

# from model_class_mnist import CNN
# from adversarial_attacks import perform_attack, determine_attack

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


def train(dataloader, model, loss_fn, optimizer, device):

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

    loss_of_epoch = loss
    return loss_of_epoch


def test(dataloader, model, loss_fn, device):

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
    return test_loss, correct


def manipulating_images_for_attack(testing_data, device):

    images, labels = [], []

    for image, label in testing_data:

        # Appending images to images list in order to use later for attacks
        images.append(image.tolist())

        # Appending labels to labels list in order to use later for attacks
        labels.append(label)

    # Sending images to either GPU or CPU
    images = torch.Tensor(images).to(device)
    labels = torch.Tensor(labels).type(torch.LongTensor).to(
        device)  # Sending labels to either GPU or CPU

    print("\nThe shape of the images list is: {}".format(images.shape))
    print("The shape of the labels list is: {}\n".format(labels.shape))
    return images, labels


def determine_attack(which_attack):
    if which_attack == "FGSM":
        return fb.attacks.LinfFastGradientAttack()
    elif which_attack == "CarliniWagner":
        return fb.attacks.L2CarliniWagnerAttack()
    elif which_attack == "PGD":
        return fb.attacks.LinfProjectedGradientDescentAttack()


def test_model_performace(samples, labels, model, device):
    size = len(samples)
    model.to(device)
    model.eval()
    correct = 0
    for sample, label in zip(samples, labels):
        sample = sample.reshape((-1, 1, 28, 28))
        sample, label = sample.to(device), label.to(device)
        pred = model(sample)
        correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    correct /= size
    classification_accuracy = correct * 100
    return classification_accuracy


def perform_attack(choose_attack, model, epsilons, images, labels, device):

    internal_data = {"attack": [],
                     "epsilon": [],
                     "classification_accuracy": []
                     }

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = determine_attack(choose_attack)
    print("\nUsing {} attack:".format(choose_attack))

    for epsilon in epsilons:
        raw_advs, clipped_advs, success = attack(
            fmodel,
            images,
            labels,
            epsilons=epsilon)

        print("\nEpsilon: {}".format(epsilon))

        result = test_model_performace(clipped_advs,
                                       labels,
                                       model,
                                       device)

        internal_data["attack"].append(choose_attack)
        internal_data["epsilon"].append(epsilon)
        internal_data["classification_accuracy"].append(result)

    df = pd.DataFrame(data=internal_data)
    print(df)

    plt.plot(df["classification_accuracy"])
    plt.title(
        "Classification Accuracy of Model for different Epsilon Values \n for Adversarial Attacks")
    plt.show()

    print("{} attack is complete.".format(choose_attack))


if __name__ == "__main__":

    batch_size = 256
    learning_rate = 0.001
    weight_decay = 0.0004
    momentum = 0.9

    epochs = 30  # Setting number of epochs for model to train

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

    # Define model - The model file is being imported from another file in the directory
    model = CNN().to(device)

    # ------------------------------------------------------------------------------------------------
    # Standard Classification Loss:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    print(model)

    training_loss_tracker = []
    validation_loss_tracker = []
    accuracy_tracker = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # Training function is defined in deep_learning_functions.py
        training_loss_of_epoch = train(
            train_dataloader, model, loss_fn, optimizer, device)
        training_loss_tracker.append(training_loss_of_epoch)
        # Test function is defined in deep_learning_functions.py
        loss_of_epoch, accuracy_of_epoch = test(
            test_dataloader, model, loss_fn, device)
        validation_loss_tracker.append(loss_of_epoch)
        accuracy_tracker.append(accuracy_of_epoch)
    print("Done!")

    plt.plot(validation_loss_tracker)
    plt.plot(training_loss_tracker)
    plt.title("Loss for each Epoch During Training")
    plt.legend()
    plt.show()

    plt.plot(accuracy_tracker)
    plt.title("Classification Accuracy for each Epoch During Training")
    plt.show()

    PATH = './mnist_dl_model.pth'
    torch.save(model.state_dict(), PATH)

    images_for_attack, labels_for_attack = manipulating_images_for_attack(
        testing_data=test_data, device=device)

    perform_attack(choose_attack="PGD",
                   model=model,
                   epsilons=[
                       0.01, 0.02],
                   images=images_for_attack,
                   labels=labels_for_attack,
                   device=device)

    perform_attack(choose_attack="FGSM",
                   model=model,
                   epsilons=[
                       0.01, 0.02],
                   images=images_for_attack,
                   labels=labels_for_attack,
                   device=device)
