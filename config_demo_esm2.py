import torch.optim as optim
import model as model  # 모델이 model.py에 있다고 가정합니다
import encoder as encoder  # 필요한 경우 encoder.py 파일에 맞춰 수정하세요
import torch
import torch.nn as nn
config = {
    "chkp_name": "ems2_model",  # 모델 체크포인트 이름
    "chkp_path": "models",  # 모델 체크포인트 저장 경로
    "log_file": "train_ems2.log",  # 학습 로그 파일 경로
    "plot_path": "plots",  # 플롯 저장 경로
    "seed": 42,  # 시드 값 설정
    "model": model.BasicTransformerBinaryClassifier,  # 새로 추가한 모델을 참조
    "model_args": {
        "input_dim": 2560,
        "num_heads": 4,
        "num_layers": 2,
        "hidden_dim": 512,
    },
    "encoder": encoder.deepneo,  # 필요시 수정
    "encoder_args": {},

    "Data": {
        "hla_embedding_path": "workspace/1025_hwangemb/1010basichla_embedding_diff1006.npy",
        "epitope_embedding_path": "workspace/1025_hwangemb/1010basicepitope_embeddings1006.npy",
        "meta_data_path": "workspace/1025_hwangemb/1010basicencoded_sequences_metadata1006.csv",
        "hla_embedding_test_path" : "workspace/1025_hwangemb/1010basichla_embedding_diff1006test.npy",
        "epitope_embedding_test_path" : "workspace/1025_hwangemb/1010basicepitope_embeddings1006test.npy",
        "meta_data_test_path" : "workspace/1025_hwangemb/1010basicencoded_sequences_metadata1006test.csv",
        "val_size": 0.3,  # 검증 데이터 비율
        "num_workers": 8,  # DataLoader의 병렬 처리 워커 수
    },

    "Train": {
        "batch_size": 128,  # 배치 사이즈
        "num_epochs": 100,  # 에포크 수
        "learning_rate": 0.002,  # 학습률
        "patience"      : 10,
        "early_stop_patience": 10,  # Early stopping patience
        "regularize": False,  # 규제 여부
        "criterion": torch.nn.BCELoss,  # 손실 함수
        "optimizer": optim.Adam,  # 옵티마이저
        "optimizer_args": {
            "lr": 0.002,  # 학습률
        },
        "use_scheduler": False,  # 스케줄러 사용 여부
    },
    
    "Test": {
        "batch_size": 128,  # 테스트 배치 사이즈
        "chkp_prefix": "best",  # 체크포인트 파일명 접두사
    },
}