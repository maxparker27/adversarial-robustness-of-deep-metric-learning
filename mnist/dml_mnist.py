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
from foolbox.criteria import Misclassification
from foolbox import attacks, models


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


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(X)
        hard_pairs = miner(outputs, y)

        loss = loss_fn(outputs, y, hard_pairs)

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings,
                                                  train_embeddings,
                                                  test_labels,
                                                  train_labels,
                                                  False)
    print(
        "Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


for epoch in range(1, num_epochs+1):
    train(train_dataloader, model, loss_fn, optimizer)
    test(training_data, test_data, model, accuracy_calculator)

PATH = './mnist_dml_model.pth'
torch.save(model.state_dict(), PATH)

model.eval()


# ------------------------------------------------------------------------------------------------
# Standard Classification Loss:
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# print(model)


# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#         optimizer.zero_grad()

#         outputs = model(X)
#         loss = loss_fn(outputs, y)

#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= size
#     correct /= size
#     print(
#         f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# epochs = 25
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")
# ------------------------------------------------------------------------------------------------
