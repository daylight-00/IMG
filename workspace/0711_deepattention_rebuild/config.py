# config.py
import os

# BASE_DIR을 현재 파일의 디렉토리로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 설정값을 포함하는 딕셔너리
config = {
    "do_train": True,
    "do_test": True,
    "Data": {
        "data_file": os.path.join(BASE_DIR, 'dataset', "transformed_dataset120240711.txt"),
        "HLA_file": os.path.join(BASE_DIR, 'dataset', "mhc_i_protein_seq2.txt"),
        "max_len_hla": 385,
        "max_len_pep": 15
    },
    "Training": {
        "epochs": 5,
        "start_lr": 0.2,
        "min_lr": 0.0001,
        "grad_clip": 0.5,
        "batch_size": 32,
        "loss_delta": 0.0001
    },
    "Paths": {
        "working_dir": "dup_0"
    },
    "Model": {
        "encoding_method": "one_hot",
        "encoding_method2": "blosum"
    },
    "model_count": 10,
    "base_model_count": 2
}
