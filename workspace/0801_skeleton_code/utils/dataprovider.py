import os
import random
import re
import torch
from torch.utils.data import Dataset, DataLoader

class DataProvider(Dataset):
    def __init__(
            self,
            epi_path,
            hla_path,
            shuffle=False):
        self.epi_path = epi_path
        self.hla_path = hla_path
        self.shuffle = shuffle

        self.hla_seq_map = self.make_hla_seq_map()
        self.samples = self.get_samples()

    def normalize_hla_name(self, hla_name):
        hla_name = re.sub(r'\*|:|-', '', hla_name)
        return hla_name

    def make_hla_seq_map(self):
        hla_seq_map = {}
        with open(self.hla_path, 'r') as in_file:
            for line in in_file:
                info = line.strip('\n').split('\t')
                hla_name = self.normalize_hla_name(info[0])
                hla_seq = info[1]
                hla_seq_map[hla_name] = hla_seq
        print(f'Number of HLA alleles: {len(hla_seq_map)}')
        return hla_seq_map

    def get_samples(self):
        samples = []
        with open(self.epi_path, 'r') as in_file:
            for line in in_file:
                info = line.strip('\n').split('\t')
                hla_name = self.normalize_hla_name(info[0])
                if hla_name not in self.hla_seq_map:
                    continue
                epi_seq = info[1]
                target = float(info[-1])
                samples.append((hla_name, epi_seq, target))
        print(f'Number of samples: {len(samples)}')
        if self.shuffle:
            random.shuffle(self.samples)
        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hla_name, epi_seq, target = self.samples[idx]
        hla_seq = self.hla_seq_map[hla_name]
        return hla_name, epi_seq, target, hla_seq