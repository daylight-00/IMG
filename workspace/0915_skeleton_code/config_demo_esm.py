import torch.nn as nn
import torch.optim as optim
import model as model       # Change here if you have a different model.py file
import encoder as encoder   # Change here if you have a different encoder.py file

config = {
    "chkp_name"         : "deepneo",
    "chkp_path"         : "models",
    "log_file"          : "../training.log",

    "model"             : model.ESM_Blosum_Base,
    "model_args"        : {
    },
    "encoder"           : encoder.esm_blosum,
    "encoder_args"      : {
        "emb_path"      : "/home/hwjang/Desktop/Research/Immunogenicity/240914/HLAseq_full_embeddings_simplified.hdf5",
    },

    "Data": {
        "epi_path"      : "dataset/deepneo/mhc1.trainset.tsv",
        "epi_args"      : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : "\t",
        },
        "hla_path"      : "dataset/deepneo/HLAseq.dat",
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "seperator" : "\t",
        },
        "test_path"     : "dataset/deepneo/mhc1.testset.tsv",
        "test_args"     : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : "\t",
        },
        "num_workers"   : 8,
        "val_size"      : 0.2,
    },

    "Train": {
        "batch_size"    : 64,
        "num_epochs"    : 100,
        "patience"      : 10,
        "regularize"    : False,            # true if regularize method is implemented in the model
        "criterion"     : nn.BCELoss,
        "optimizer"     : optim.SGD,
        "optimizer_args": {
            "lr"        : 0.01,
            "momentum"  : 0.9,
        }
    },
    
    "Test": {
        "batch_size"    : 64,
        "chkp_prefix"   : "best",
    },
}