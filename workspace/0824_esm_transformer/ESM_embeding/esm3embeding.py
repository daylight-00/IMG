import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ESM3 모델 불러오기 with the device correctly set
client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)

# 입력 파일 경로 및 출력 파일 경로 지정
input_file_path = 'HLAseq_remove_msa.dat'
output_file_path = 'HLAseq_embeddings.pt'

# 파일 읽기
with open(input_file_path, 'r') as f:
    sequences = f.read().splitlines()

# 각 단백질 서열에 대한 임베딩 계산 및 저장
embeddings = []
for sequence in sequences:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    
    output = client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
    embeddings.append(output.per_residue_embedding.cpu())  # CUDA 장치에서 CPU로 이동

# 결과를 텐서 파일로 저장
torch.save(embeddings, output_file_path)

print(f"Embeddings have been saved to {output_file_path}")
