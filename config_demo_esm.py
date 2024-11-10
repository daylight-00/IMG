import torch.nn as nn
import torch.optim as optim
import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file

config = {
    "chkp_name"         : "demo_esm_blo",
    "chkp_path"         : "models",
    "log_file"          : "train.log",
    "plot_path"         : "plots",
    "seed"              : 100,

    "model"             : model.Cross_Attn_Demo,
    "model_args"        : {
        "hla_dim"       : 1536,
        "epi_dim"       : 25,
        "hla_nhead"     : 8,
        "epi_nhead"     : 5,
    },
    "encoder"           : encoder.plm_blosum,
    "encoder_args"      : {
        "hla_emb_path"  : "/home/public/project/EMB/emb-hla1-esm3-pad-1105.h5",
    },

    "Data": {
        "epi_path"      : "/home/public/project/IMG/data/deepneo/mhc1.trainset.csv",
        "epi_args"      : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : ",",
        },
        "hla_path"      : "/home/public/project/IMG/data/deepneo/HLAseq.csv",
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "seperator" : ",",
        },
        "test_path"     : "/home/public/project/IMG/data/deepneo/mhc1.testset.csv",
        "test_args"     : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : ",",
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
        },
        "use_scheduler" : False,
    },
    
    "Test": {
        "batch_size"    : 64,
        "chkp_prefix"   : "best",
    },
}