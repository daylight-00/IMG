# config.py
import os

# BASE_DIR을 현재 파일의 디렉토리로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 설정값을 포함하는 딕셔너리
config = {
    "do_train": True,
    "do_test": True,
    "Data": {
        "data_file": os.path.join(BASE_DIR, 'dataset', "/home/alex/IMG/workspace/0727_deepneo_rebuild/dataset/mhc1.trainset.tsv"),
        "HLA_file": os.path.join(BASE_DIR, 'dataset', "/home/alex/IMG/workspace/0727_deepneo_rebuild/dataset/HLAseq.dat"),
        "max_len_hla": 369,
        "max_len_pep": 9
    },

}
