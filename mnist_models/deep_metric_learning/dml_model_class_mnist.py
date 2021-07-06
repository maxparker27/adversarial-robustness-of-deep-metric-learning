from torch import nn
import torch.nn.functional as F


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
