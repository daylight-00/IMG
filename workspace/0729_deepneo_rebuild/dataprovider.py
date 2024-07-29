import os
import random
import re
import torch
from torch.utils.data import Dataset, DataLoader
from config import config

class DataProvider(Dataset):
    def __init__(self, data_file=config["Data"]["data_file"], hla_file=config["Data"]["HLA_file"], shuffle=False):
        self.data_file = data_file
        self.hla_file = hla_file
        self.shuffle = shuffle
        self.hla_sequence = {}
        self.read_hla_sequences()
        self.samples = []
        self.read_training_data()

    def normalize_hla_name(self, hla_name):
        hla_name = re.sub(r'\*|:|-', '', hla_name)
        return hla_name

    def read_hla_sequences(self):
        with open(self.hla_file, 'r') as in_file:
            for line in in_file:
                info = line.strip('\n').split('\t')
                info[0] = self.normalize_hla_name(info[0])
                seq = info[1]
                self.hla_sequence[info[0]] = seq

    def read_training_data(self):
        with open(self.data_file, 'r') as in_file:
            for line in in_file:
                info = line.strip('\n').split('\t')
                hla_a = self.normalize_hla_name(info[0])
                if hla_a not in self.hla_sequence:
                    continue
                peptide = info[1]
                affinity = float(info[-1])
                self.samples.append((hla_a, peptide, affinity))

        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hla_a, peptide, affinity = self.samples[idx]
        hla_seq = self.hla_sequence[hla_a]
        return hla_a, peptide, affinity, hla_seq
