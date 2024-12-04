import torch.nn as nn
import torch.optim as optim
import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file

config = {
    "chkp_name"         : "demo_deepneo",
    "chkp_path"         : "models",
    "log_file"          : "train.log",
    "plot_path"         : "plots",
    "seed"              : 100,
    "model"             : model.DeepNeo_2_Custom,
    "model_args"        : {
    },
    "encoder"           : encoder.deepneo_2,
    "encoder_args"      : {
    },

    "Data": {
        "epi_path"      : "/home/public/project/IMG/data/deepneo_2/mhc2.trainset.csv",
        "epi_args"      : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : ",",
        },
        "hla_path"      : "/home/public/project/IMG/data/deepneo_2/MHC2_prot_alignseq_unique.csv",
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "seperator" : ",",
        },
        "test_path"     : "/home/public/project/IMG/data/deepneo_2/mhc2.testset.csv",
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
        "batch_size"    : 1024,
        "num_epochs"    : 100,
        "patience"      : 10,
        "regularize"    : True,            # true if regularize method is implemented in the model
        "criterion"     : nn.BCELoss,
        "optimizer"     : optim.SGD,
        "optimizer_args": {
            "lr"        : 0.01,
            "momentum"  : 0.9,
        },
        "use_scheduler" : False,
    },
    
    "Test": {
        "batch_size"    : 1024,
        "chkp_prefix"   : "best",
    },
}