import torch.nn.functional as F
import torch.nn as nn
import torch

class TestModel(nn.Module):
    def __init__(self):
        super(DeepNeo, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(5, 183), stride=1)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=10, kernel_size=(5, 183), stride=1)
        self.fc1 = nn.Linear(1 * 5 * 10, 50)
        self.fc2 = nn.Linear(50, 128)
        self.fc3 = nn.Linear(128, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class DeepNeo(nn.Module):
    def __init__(self):
        super(DeepNeo, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(5, 183), stride=1)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=10, kernel_size=(5, 183), stride=1)
        self.fc = nn.Linear(1*5*10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc(x))
        x = torch.sigmoid(x)
        return x

model_map = {
    'test'     : TestModel(),
    'DeepNeo'   : DeepNeo(),
}