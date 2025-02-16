import torch.nn as nn
import torch.optim as optim
import model as model                       # Change here if you have a different `model.py` file
import encoder as encoder                   # Change here if you have a different `encoder.py` file

config = {
    "chkp_name"         : "esmc_small",
    "chkp_path"         : "models",
    "log_file"          : "train.log",
    "plot_path"         : "plots",
    "seed"              : 128,

    "model"             : model.plm_cat_mean,
    "model_args"        : {
        "hla_dim_s"       : 960,
        "hla_dim_p"       : 0,
        "epi_dim_s"       : 960,
        "epi_dim_p"       : 0,
        "head_div"        : 64,
    },

    "encoder"           : encoder.plm_plm_mask,
    "encoder_args"      : {
        "hla_emb_path_s" : "/home/alpha/project/EMB/emb_hla2_esmc_small_light_0203.h5",
        "epi_emb_path_s" : "/home/alpha/project/EMB/emb_epi_esmc_small_0203.h5",
        # "hla_emb_path_p" : "/home/alpha/project/EMB/side/emb_hla2_chai_msa_pair_light_0127_side.h5",
        # "epi_emb_path_p" : "/home/alpha/project/EMB/side/emb_epi_chai_pair_0127_side.h5",
    },
    "CrossValidation": {
        "num_folds"     : 5,
    },

    "Data": {
        "epi_path"      : "/home/alpha/project/IMG/data/final/mhc2_full_human_train.csv",
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
        "test_path"     : "/home/alpha/project/IMG/data/final/mhc2_full_human_test.csv",
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