import torch.nn as nn
import torch.optim as optim

config = {
    "model_name"        : "DeepNeo",
    "chkp_name"         : "deepneo",
    "chkp_path"         : "models",
    "do_train"          : True,
    "do_test"           : True,
    "Data": {
        "epi_path"      : "dataset/mhc1.trainset.tsv",
        "hla_path"      : "dataset/HLAseq.dat",
        "max_len_hla"   : 369,
        "max_len_pep"   : 9,
        "val_size"      : 0.2,
    },
    "Train": {
        "batch_size"    : 1024,
        "num_epochs"    : 100,
        "patience"      : 10,
        "learning_rate" : 0.01,
        "log_file"      : "log_train.txt",
        "regularize"    : True,
        "criterion"     : nn.BCELoss(),
        "optimizer"     : optim.SGD,
        "optimizer_args": {
            "lr"        : 0.01,
            "momentum"  : 0.9,
        }
    },
    "Test": {
        "batch_size"    : 1024,
        "num_workers"   : 4,
        "log_file"      : "log.txt",
        "test_set_path" : "dataset/mhc1.testset.tsv",
        "chkp_prefix"   : "best",
    },
}