import gzip
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

def Load_data(dataset):
    print('... loading data')
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    train_set_x, train_set_y = torch.tensor(train_set[0], dtype=torch.float32), torch.tensor(train_set[1], dtype=torch.long)
    valid_set_x, valid_set_y = torch.tensor(valid_set[0], dtype=torch.float32), torch.tensor(valid_set[1], dtype=torch.long)
    test_set_x, test_set_y = torch.tensor(test_set[0], dtype=torch.float32), torch.tensor(test_set[1], dtype=torch.long)

    train_dataset = TensorDataset(train_set_x, train_set_y)
    valid_dataset = TensorDataset(valid_set_x, valid_set_y)
    test_dataset = TensorDataset(test_set_x, test_set_y)

    return train_dataset, valid_dataset, test_dataset

def Load_data_ind(dataset):
    print('... loading data')
    with gzip.open(dataset, 'rb') as f:
        test_set = pickle.load(f, encoding='latin1')
    test_set_x, test_set_y = torch.tensor(test_set[0], dtype=torch.float32), torch.tensor(test_set[1], dtype=torch.long)
    test_dataset = TensorDataset(test_set_x, test_set_y)
    return test_dataset

def Load_npdata(dataset):
    print('... loading data')
    datasets = np.load(dataset)
    test_setx = datasets['test_seq']
    test_sety = datasets['test_lab']
    test_set_x, test_set_y = torch.tensor(test_setx, dtype=torch.float32), torch.tensor(test_sety, dtype=torch.long)
    test_dataset = TensorDataset(test_set_x, test_set_y)
    return test_dataset
