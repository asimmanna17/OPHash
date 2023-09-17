import torch
import torch.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hash_code):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, hash_code)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x, indices1 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)
        x = F.relu(self.conv2(x))
        x, indices2 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x, indices3 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x_c = F.relu(self.fc2(x))#x
        x_out = self.fc3(x_c)
        x_out =torch.tanh(x_out)#h
        return x_out, x_c  # h, x

class Classifier(nn.Module):
    def __init__(self, numclasses):
        super(Classifier, self).__init__()
        # self.conv1d = nn.Conv1d(2,1,1)
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, numclasses)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class Discriminator(nn.Module):
    def __init__(self, zsize):
        super(Discriminator, self).__init__()
        self.conv1d = nn.Conv1d(2, 1, 1)
        self.fc1 = nn.Linear(zsize, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1d(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Hasher(nn.Module):
    def __init__(self, zsize):
        super(Hasher, self).__init__()
        # self.conv1d = nn.Conv1d(2,1,1)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, zsize)

    def forward(self, x):
        # x = F.relu(self.conv1d(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

