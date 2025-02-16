import torch
import numpy as np
from torch.utils.data import Dataset
import h5py

#%% PLM
def get_plm_emb(emb_dict, key, max_len):
    embedding = np.squeeze(emb_dict[key][()])
    embedding = torch.tensor(embedding, dtype=torch.float32)
    emb_dim = embedding.shape[-1]
    seq_len = embedding.shape[0]
    if seq_len < max_len:
        embedding = torch.cat([embedding, torch.zeros(max_len - seq_len, emb_dim)], dim=0)
    else:
        embedding = embedding[:max_len]
    return embedding

def get_padding_mask(sequence, max_len):
    seq_len = len(sequence)
    # if seq_len == max_len:
    #     return None
    pad_mask = torch.zeros(max_len)
    pad_mask[seq_len:] = 1
    return pad_mask

class plm_plm_mask(Dataset):
    def __init__(self, data_provider, hla_emb_path_s, epi_emb_path_s, hla_emb_path_p=None, epi_emb_path_p=None):
        self.data_provider = data_provider
        self.hdf5_path_s1 = hla_emb_path_s
        self.hdf5_path_s2 = epi_emb_path_s
        self.hdf5_path_p1 = hla_emb_path_p
        self.hdf5_path_p2 = epi_emb_path_p
        self.hla_emb_dict_s = h5py.File(self.hdf5_path_s1, 'r', libver='latest')
        self.epi_emb_dict_s = h5py.File(self.hdf5_path_s2, 'r', libver='latest')
        self.hla_emb_dict_p = h5py.File(self.hdf5_path_p1, 'r', libver='latest') if hla_emb_path_p is not None else None
        self.epi_emb_dict_p = h5py.File(self.hdf5_path_p2, 'r', libver='latest') if epi_emb_path_p is not None else None

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, target, hla_seq = self.data_provider[idx]
        # Load embeddings
        hla_emb_s = get_plm_emb(self.hla_emb_dict_s, hla_name, 269)
        epi_emb_s = get_plm_emb(self.epi_emb_dict_s, epi_seq, 15)
        hla_emb_p = get_plm_emb(self.hla_emb_dict_p, hla_name, 269) if self.hla_emb_dict_p is not None else False
        epi_emb_p = get_plm_emb(self.epi_emb_dict_p, epi_seq, 15) if self.epi_emb_dict_p is not None else False
        # Create padding masks
        pad_mask_hla = get_padding_mask(hla_seq, 269)
        pad_mask_epi = get_padding_mask(epi_seq, 15)
        # Convert target to tensor
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return hla_emb_s, hla_emb_p, epi_emb_s, epi_emb_p, pad_mask_hla, pad_mask_epi, target

    def __del__(self):
        self.hla_emb_dict_s.close()
        self.epi_emb_dict_s.close()
        self.hla_emb_dict_p.close() if self.hla_emb_dict_p is not None else None
        self.epi_emb_dict_p.close() if self.epi_emb_dict_p is not None else None

#%% BLOSUM
def get_blosum_emb(matrix, sequence, max_len):
    embedding = [matrix[aa] for aa in sequence]
    embedding = torch.tensor(embedding, dtype=torch.float32)
    emb_dim = embedding.shape[-1]
    seq_len = embedding.shape[0]
    if seq_len < max_len:
        embedding = torch.cat([embedding, torch.zeros(max_len - seq_len, emb_dim)], dim=0)
    else:
        embedding = embedding[:max_len]
    return embedding

class blosum_mask(Dataset):
    def __init__(self, data_provider, hla_emb_path_s=None, epi_emb_path_s=None, hla_emb_path_p=None, epi_emb_path_p=None):
        self.data_provider = data_provider
        from utils.matrix import blosum62
        self.blosum62 = blosum62

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, target, hla_seq = self.data_provider[idx]
        # Load embeddings
        hla_emb_s = get_blosum_emb(self.blosum62, hla_seq, 269)
        epi_emb_s = get_blosum_emb(self.blosum62, epi_seq, 15)
        hla_emb_p = False
        epi_emb_p = False
        # Create padding masks
        pad_mask_hla = get_padding_mask(hla_seq, 269)
        pad_mask_epi = get_padding_mask(epi_seq, 15)
        # Convert target to tensor
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return hla_emb_s, hla_emb_p, epi_emb_s, epi_emb_p, pad_mask_hla, pad_mask_epi, target

#%% DEEPNEO
class deepneo(Dataset):
    def __init__(self, data_provider, matrix_size=(15, 269)):
        self.data_provider = data_provider
        self.matrix_size = matrix_size
        from utils.matrix import get_calpha_matrix
        self.get_calpha_matrix = get_calpha_matrix

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, target, hla_seq = self.data_provider[idx]
        encoded_matrix = self.get_calpha_matrix(hla_seq, epi_seq, self.matrix_size)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        encoded_matrix = torch.tensor(encoded_matrix, dtype=torch.float32).unsqueeze(0)
        return encoded_matrix, target