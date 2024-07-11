import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from load_data import Load_data
from pyroc import *

class LeNetConvPoolLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(LeNetConvPoolLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class LogisticRegression(nn.Module):
    def __init__(self, n_in, n_out):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x)).flatten()
        return x

class CNN(nn.Module):
    def __init__(self, in_dim, nkerns, filtsize, poolsize, hidden):
        super(CNN, self).__init__()
        self.layer0 = LeNetConvPoolLayer(1, nkerns[0], filtsize[0], poolsize[0])
        self.layer1 = LeNetConvPoolLayer(nkerns[0], nkerns[1], filtsize[1], poolsize[1])
        
        dim11 = in_dim[0] - filtsize[0][0] + 1
        dim12 = in_dim[1] - filtsize[0][1] + 1
        dim21 = dim11 - filtsize[1][0] + 1
        dim22 = dim12 - filtsize[1][1] + 1
        
        self.layer2_input_dim = nkerns[1] * dim21 * dim22
        self.layer2 = LogisticRegression(self.layer2_input_dim, hidden)

    def forward(self, x):
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        x = self.layer0(x)
        x = self.layer1(x)
        x = x.view(-1, self.layer2_input_dim)
        x = self.layer2(x)
        return x

    def negative_log_likelihood(self, y, output):
        return -torch.mean(y * torch.log(output) + (1 - y) * torch.log(1. - output))

    def build_finetune_functions(self, datasets, batch_size, learning_rate, L1_param, L2_param, mom):
        train_dataset, valid_dataset, test_dataset = datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=mom)

        def train_fn():
            self.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self(x_batch)
                loss = self.negative_log_likelihood(y_batch.float(), output)
                loss.backward()
                optimizer.step()
            return loss.item()

        def valid_check():
            self.eval()
            valid_y, valid_pred = [], []
            with torch.no_grad():
                for x_batch, y_batch in valid_loader:
                    output = self(x_batch)
                    valid_y.extend(y_batch.tolist())
                    valid_pred.extend(output.tolist())
            return valid_y, valid_pred

        def test_check():
            self.eval()
            test_y, test_pred = [], []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    output = self(x_batch)
                    test_y.extend(y_batch.tolist())
                    test_pred.extend(output.tolist())
            return test_y, test_pred

        return train_fn, valid_check, test_check
