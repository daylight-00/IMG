import torch.nn as nn
import torch.optim as optim
import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file

config = {
    "chkp_name"         : "af3",
    "chkp_path"         : "models",
    "log_file"          : "train.log",
    "plot_path"         : "plots",
    "seed"              : 128,

    "model"             : model.plm_cat_mean,
    "model_args"        : {
        "hla_dim_s"       : 384,
        "hla_dim_p"       : 256,
        "epi_dim_s"       : 384,
        "epi_dim_p"       : 256,
        "head_div"        : 64,
    },

    "encoder"           : encoder.plm_plm_mask,
    "encoder_args"      : {
        "hla_emb_path_s" : "/home/alpha/project/EMB/emb_hla2_af3_single_light_0127.h5",
        "epi_emb_path_s" : "/home/alpha/project/EMB/emb_epi_af3_single_0127.h5",
        "hla_emb_path_p" : "/home/alpha/project/EMB/side/emb_hla2_af3_pair_light_0127_side.h5",
        "epi_emb_path_p" : "/home/alpha/project/EMB/side/emb_epi_af3_pair_0127_side.h5",
    },
    "CrossValidation": {
        "num_folds"     : 5,
    },

    "Data": {
        "epi_path"      : "/home/alpha/project/IMG/data/250303/train.csv",
        "epi_args"      : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : ",",
        },
        "hla_path"      : "/home/alpha/project/IMG/data/imgt_msa/HLA2_IMGT_MSA_light_clean.csv",
        "hla_args"      : {
            "hla_header": 'HLA_Name',
            "seq_header": 'HLA_Seq',
            "seperator" : ",",
        },
        "test_path"     : "/home/alpha/project/IMG/data/250303/test.csv",
        "test_args"     : {
            "epi_header": 'Epi_Seq',
            "hla_header": 'HLA_Name',
            "tgt_header": 'Target',
            "seperator" : ",",
        },
        "num_workers"   : 4,
        "val_size"      : 0.2,
    },

    "Train": {
        "batch_size"    : 128,
        "num_epochs"    : 100,
        "patience"      : 10,
        "regularize"    : False,            # true if regularize method is implemented in the model
        "criterion"     : nn.BCEWithLogitsLoss,
        "optimizer"     : optim.AdamW,
        "optimizer_args": {
            "lr"        : 1e-4,
        },
        "use_scheduler" : False,
    },
    
    "Test": {
        "batch_size"    : 64,
        "chkp_prefix"   : "best",
        "plot"          : True,
        "feat_extract"  : False,
        "feat_path"     : "feat_extract",
        "target_layer"  : "output_layer",
        "save_pred"     : False,
    },
}