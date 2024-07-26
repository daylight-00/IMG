import os
import random
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import config
from encoding import ENCODING_METHOD_MAP

class DataProvider(Dataset):
    def __init__(self, data_file=config["Data"]["data_file"], hla_file=config["Data"]["HLA_file"], 
                 max_len_hla=config["Data"]["max_len_hla"], max_len_pep=config["Data"]["max_len_pep"], 
                 encoding_method='blosum', shuffle=True):
        self.data_file = data_file
        self.hla_file = hla_file
        self.max_len_hla = max_len_hla
        self.max_len_pep = max_len_pep
        self.encoding_method = encoding_method
        self.shuffle = shuffle
        self.hla_sequence = {}
        self.read_hla_sequences()
        self.apply_mutations()
        self.samples = []
        self.read_training_data()

    def normalize_hla_name(self, hla_name):
        # HLA 이름 정규화
        hla_name = re.sub(r'\*|:|-', '', hla_name)
        return hla_name

    def parse_hla_and_mutation(self, hla_with_mutation):
        parts = hla_with_mutation.split()
        hla_name = self.normalize_hla_name(parts[0])
        mutations = [mutation for mutation in parts[1:] if self.validate_mutation(mutation)]
        return hla_name, mutations

    def validate_mutation(self, mutation):
        return bool(re.match(r'^[A-Z]\d+[A-Z]$', mutation))

    def apply_mutation(self, sequence, mutation):
        pos = int(mutation[1:-1]) - 1
        new_aa = mutation[-1]
        mutated_sequence = sequence[:pos] + new_aa + sequence[pos + 1:]
        return mutated_sequence

    def read_hla_sequences(self):
        with open(self.hla_file, 'r') as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split(' ')
                hla_name = self.normalize_hla_name(info[0])
                seq = info[1]
                
                if len(seq) >= self.max_len_hla:
                    seq = seq[:self.max_len_hla]

                self.hla_sequence[hla_name] = seq

    def apply_mutations(self):
        df = pd.read_csv(self.data_file)
        for _, row in df.iterrows():
            hla_name_with_mutation = row['HLA']
            hla_name, mutations = self.parse_hla_and_mutation(hla_name_with_mutation)
            if mutations and hla_name in self.hla_sequence:
                mutated_seq = self.hla_sequence[hla_name]
                for mutation in mutations:
                    mutated_seq = self.apply_mutation(mutated_seq, mutation)
                mutant_hla_name = f"{hla_name}-{'-'.join(mutations)}"
                self.hla_sequence[mutant_hla_name] = mutated_seq

    def read_training_data(self):
        df = pd.read_csv(self.data_file)
        for _, row in df.iterrows():
            hla_a, mutations = self.parse_hla_and_mutation(row['HLA'])
            
            if hla_a not in self.hla_sequence:
                continue

            peptide = row['Peptide']
            if len(peptide) > self.max_len_pep:
                continue

            affinity = float(row['Affinity'])

            self.samples.append((hla_a, peptide, affinity))

        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hla_a, peptide, affinity = self.samples[idx]
        hla_seq = self.hla_sequence[hla_a]

        # Use the selected encoding method
        encode_fn = ENCODING_METHOD_MAP[self.encoding_method]
        hla_encoded, hla_mask = encode_fn(hla_seq, self.max_len_hla)
        pep_encoded, pep_mask = encode_fn(peptide, self.max_len_pep)
        
        return (hla_encoded, hla_mask, pep_encoded, pep_mask), affinity

    def get_first_hla_sequence(self):
        if self.hla_sequence:
            for hla, seq in self.hla_sequence.items():
                return hla, seq
        return None


# Test the DataProvider
def main():
    data_provider = DataProvider()
    dataloader = DataLoader(data_provider, batch_size=config["Training"]["batch_size"], shuffle=True)
    
    for batch in dataloader:
        (hla_encoded, hla_mask, pep_encoded, pep_mask), affinities = batch
        print(hla_encoded.shape, hla_mask.shape, pep_encoded.shape, pep_mask.shape)
        break

    # Save HLA sequences to CSV
    hla_sequences = data_provider.hla_sequence
    hla_df = pd.DataFrame(list(hla_sequences.items()), columns=['HLA', 'Sequence'])
    hla_df.to_csv('hla_sequences.csv', index=False)
    print("HLA sequences have been saved to hla_sequences.csv")

if __name__ == "__main__":
    main()
