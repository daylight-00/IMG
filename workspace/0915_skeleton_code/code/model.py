import torch.nn.functional as F
import torch.nn as nn
import torch
#%%

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(5, 183), stride=1)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=10, kernel_size=(5, 183), stride=1)
        self.fc1 = nn.Linear(1 * 5 * 10, 50)
        self.fc2 = nn.Linear(50, 128)
        self.fc3 = nn.Linear(128, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

#%%
class DeepNeo(nn.Module):
    def __init__(self):
        super(DeepNeo, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(5, 183), stride=1)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=10, kernel_size=(5, 183), stride=1)
        self.fc = nn.Linear(1*5*10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc(x))
        x = torch.sigmoid(x)
        return x
    
    def regularize(self, loss, device):
        l1_lambda = 0.0001
        l2_lambda = 0.001
        l2_reg = torch.tensor(0.).to(device)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        loss += l2_lambda*l2_reg + l1_lambda*torch.norm(self.fc.weight, 1)
        return loss
    
#%%
class HLAEmbeddingProcessor(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):  # 1536차원에서 8개 헤드 사용
        super(HLAEmbeddingProcessor, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=0.2)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embedding_tensor):
        # HLA 임베딩을 Self-Attention으로 처리
        embedding_tensor = embedding_tensor.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        attn_output, _ = self.self_attention(embedding_tensor, embedding_tensor, embedding_tensor)
        attn_output = self.layer_norm(attn_output)
        
        # 시퀀스 차원 평균화로 seq_len 제거
        attn_output = attn_output.mean(dim=0)  # (batch_size, embedding_dim)
        return attn_output

class ESM_Blosum_Base(nn.Module):
    def __init__(self, input_dim=1536, d_model=25, nhead=5, num_layers=10, dropout_rate=0.2):
        super(ESM_Blosum_Base, self).__init__()

        # HLA embedding processor with self-attention
        self.hla_processor = HLAEmbeddingProcessor(embedding_dim=input_dim, num_heads=8)  # 1536차원에서 8개 헤드 사용

        # Epitope 인코딩의 Self-Attention 레이어 (25차원)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_rate)

        # Fully connected layers to reduce HLA embedding dimensions to 25
        self.fc_query = nn.Linear(input_dim, d_model)
        self.fc_key = nn.Linear(input_dim, d_model)
        self.fc_value = nn.Linear(input_dim, d_model)

        # MultiheadAttention layers with 25 dimensions
        self.multihead_attn_1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_rate)
        self.multihead_attn_2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_rate)
        
        
        self.self_attention_2 = nn.MultiheadAttention(embed_dim=50, num_heads=nhead, dropout=dropout_rate)
        #self.self_attention_3 = nn.MultiheadAttention(embed_dim=50, num_heads=nhead, dropout=dropout_rate)

        # Layer normalization and dropout after each self-attention layer
        self.layer_norm_1 = nn.LayerNorm(50)
        #self.layer_norm_2 = nn.LayerNorm(50)
 
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Output layer for final prediction
        self.output_layer = nn.Linear(50, 1)

    def forward(self, hla_embedding, epitope_encoded):
        # HLA 임베딩을 Self-Attention 후 평균화 (1536차원)
        hla_embedding = self.hla_processor(hla_embedding)  # (batch_size, 1536)

        # Epitope 인코딩의 Self-Attention (25차원)
        epitope_encoded = epitope_encoded.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        attn_output, _ = self.self_attention(epitope_encoded, epitope_encoded, epitope_encoded)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, d_model)

        # 시퀀스 차원 평균화로 seq_len 제거
        query1 = attn_output.mean(dim=1)  # (batch_size, d_model)
        key2 = attn_output.mean(dim=1)
        value2 = attn_output.mean(dim=1)
        
        # HLA 임베딩을 25차원으로 축소
        key1 = self.fc_key(hla_embedding)  # (batch_size, d_model)
        value1 = self.fc_value(hla_embedding)  # (batch_size, d_model)
        query2 = self.fc_query(hla_embedding)  # (batch_size, d_model)

        # Reshape query, key, value for MultiheadAttention with 25 dimensions
        query1= query1.unsqueeze(0)  # (1, batch_size, d_model)
        key1 = key1.unsqueeze(0)  # (1, batch_size, d_model)
        value1 = value1.unsqueeze(0)  # (1, batch_size, d_model)

        query2 = query2.unsqueeze(0)  # (1, batch_size, d_model)
        key2 = key2.unsqueeze(0)  # (1, batch_size, d_model)
        value2 = value2.unsqueeze(0)  # (1, batch_size, d_model)

        # MultiheadAttention with 25 dimensions
        attn_output1, _ = self.multihead_attn_1(query1, key1, value1)  # (1, batch_size, d_model)
        attn_output2, _ = self.multihead_attn_2(query2, key2, value2)  # (1, batch_size, d_model)

        # Concatenate attn_output1 and attn_output2 along the feature dimension
        attn_output = torch.cat((attn_output1, attn_output2), dim=-1)  # (1, batch_size, 2*d_model)

        # 추가 Self-Attention 레이어 적용 (25차원)
        attn_output, _ = self.self_attention_2(attn_output, attn_output, attn_output)
        attn_output = self.layer_norm_1(attn_output)

        #attn_output, _ = self.self_attention_3(attn_output, attn_output, attn_output)
        #attn_output = self.layer_norm_2(attn_output)

        # seq_len 차원 제거
        attn_output = attn_output.squeeze(0)  # (batch_size, d_model)

        # 드롭아웃 적용
        attn_output = self.dropout(attn_output)

        # 최종 출력 레이어 + 시그모이드 활성화 함수
        output = torch.sigmoid(self.output_layer(attn_output))

        return output