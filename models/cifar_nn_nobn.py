import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar_nn_nobn(nn.Module):
    def __init__(self, input_shape: tuple = (128, 3, 32, 32)):
        super().__init__()
        self.return_softmax = False

        self.probDropout = 0.5
        self.conv1_1 = nn.LazyConv2d(64, 3)

        self.conv1_2 = nn.LazyConv2d(64, 3)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.LazyConv2d(128, 3)
        self.conv2_2 = nn.LazyConv2d(128, 3)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.LazyLinear(256)
        self.dropFc1 = nn.Dropout(p=self.probDropout)
        self.fc2 = nn.LazyLinear(256)

        self.dropFc2 = nn.Dropout(p=self.probDropout)
        self.fc3 = nn.LazyLinear(10)
        self.softmaxLayer = nn.Softmax(dim=1)
        #  Perform dry run for parameter initialization:
        self.forward(torch.ones(input_shape))

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        x = self.pool1(x)


        x = F.relu(self.conv2_1(x)) # No batchnorm
        x = F.relu(self.conv2_2(x))

        x = self.pool2(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = self.dropFc1(F.relu(self.fc1(x)))
        x = self.dropFc2(F.relu(self.fc2(x)))

        x = self.fc3(x)
        z = self.softmaxLayer(x)

        return x
