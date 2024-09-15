import torch.nn as nn
import torch.optim as optim
import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file

config = {
    "chkp_name"         : "deepneo",
    "chkp_path"         : "models",
    "log_file"          : "train.log",
    "plot_path"         : "plots",

    "model"             : model.DeepNeo,
    "model_args"        : {
    },
    "encoder"           : encoder.deepneo,
    "encoder_args"      : {
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
        "batch_size"    : 1024,
        "num_epochs"    : 100,
        "patience"      : 10,
        "regularize"    : True,            # true if regularize method is implemented in the model
        "criterion"     : nn.BCELoss,
        "optimizer"     : optim.SGD,
        "optimizer_args": {
            "lr"        : 0.01,
            "momentum"  : 0.9,
        }
    },
    
    "Test": {
        "batch_size"    : 1024,
        "chkp_prefix"   : "best",
    },
}