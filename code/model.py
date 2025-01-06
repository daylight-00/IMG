import torch.nn.functional as F
import torch.nn as nn
import torch

def find_optimal_nhead(self, concat_dim):
    for nhead in range(10, 0, -1):
        if concat_dim % nhead == 0:
            return nhead
    return 1

class _simple_self_attn_block(nn.Module):
    def __init__(self, embed_dim, nhead, dropout):
        super(_simple_self_attn_block, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # seq_len, batch_size, embed_dim
        attn_output, _ = self.self_attn(x, x, x)
        x = attn_output + x  # Residual connection
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x

class _ffn_residual_self_attn_block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(_ffn_residual_self_attn_block, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)    
        x = self.layer_norm2(x + self.dropout(ffn_output))
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        return x
    
class simple_self_attn(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, n_blocks=1):
        super(simple_self_attn, self).__init__()
        self.blocks = nn.ModuleList([
            _ffn_residual_self_attn_block(embed_dim, num_heads, dropout) for _ in range(n_blocks)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class simple_cross_attn(nn.Module):
    def __init__(self, embed_dim, nhead, dropout):
        super(simple_cross_attn, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x1, x2):
        x1 = x1.permute(1, 0, 2)  # seq_len, batch_size, embed_dim
        x2 = x2.permute(1, 0, 2)
        attn_output, _ = self.cross_attn(x1, x2, x2)
        x1 = attn_output + x1  # Residual connection
        x1 = self.layer_norm(x1)
        x1 = self.dropout(x1)
        x1 = x1.permute(1, 0, 2)
        return x1

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
class DeepNeo_2_Custom(nn.Module):
    def __init__(self):
        super(DeepNeo_2_Custom, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(8, 133), stride=1)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=10, kernel_size=(8, 133), stride=1)
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
class HLA_Preproc(nn.Module):
    def __init__(self, hla_dim, epi_dim, nhead, dropout_rate):
        super(HLA_Preproc, self).__init__()
        self.hla_self_attn = nn.MultiheadAttention(embed_dim=hla_dim, num_heads=nhead, dropout=dropout_rate)
        self.hla_layer_norm = nn.LayerNorm(hla_dim)
        if hla_dim > epi_dim:
            self.proj = True
            self.hla_q_proj = nn.Linear(hla_dim, epi_dim)
            self.hla_k_proj = nn.Linear(hla_dim, epi_dim)
            self.hla_v_proj = nn.Linear(hla_dim, epi_dim)
        else:
            self.proj = False

    def forward(self, x_hla):
        # Input shape: (batch_size, seq_len, hla_dim)
        x_hla = x_hla.permute(1, 0, 2)  # (seq_len, batch_size, hla_dim)
        x_hla, _ = self.hla_self_attn(x_hla, x_hla, x_hla)
        x_hla = self.hla_layer_norm(x_hla)
        x_hla = x_hla.permute(1, 0, 2)  # (batch_size, seq_len, hla_dim)
        x_hla = x_hla.mean(dim=1)  # (batch_size, hla_dim)
        if self.proj:
            Q = self.hla_q_proj(x_hla).unsqueeze(0)  # (1, batch_size, epi_dim)
            K = self.hla_k_proj(x_hla).unsqueeze(0)
            V = self.hla_v_proj(x_hla).unsqueeze(0)
        else:
            Q = x_hla.unsqueeze(0)    # (1, batch_size, hla_dim)
            K = x_hla.unsqueeze(0)
            V = x_hla.unsqueeze(0)
        return Q, K, V

class Epi_Preproc(nn.Module):
    def __init__(self, hla_dim, epi_dim, nhead, dropout_rate):
        super(Epi_Preproc, self).__init__()
        self.epi_self_attn = nn.MultiheadAttention(embed_dim=epi_dim, num_heads=nhead, dropout=dropout_rate)

    def forward(self, x_epi):
        # Input shape: (batch_size, seq_len, epi_dim)
        x_epi = x_epi.permute(1, 0, 2)  # (seq_len, batch_size, epi_dim)
        x_epi, _ = self.epi_self_attn(x_epi, x_epi, x_epi)
        x_epi = x_epi.permute(1, 0, 2)  # (batch_size, seq_len, epi_dim)
        x_epi = x_epi.mean(dim=1)  # (batch_size, epi_dim)

        Q = x_epi.unsqueeze(0)     # (1, batch_size, epi_dim)
        K = x_epi.unsqueeze(0)
        V = x_epi.unsqueeze(0)
        return Q, K, V

class Cross_Attn_Demo(nn.Module):
    def __init__(self, hla_dim=1536, epi_dim=25, hla_nhead=8, epi_nhead=5, d_model=1536, dropout_rate=0.2):
        super(Cross_Attn_Demo, self).__init__()
        proj_dim = epi_dim
        self.hla_preprocessor = HLA_Preproc(hla_dim=hla_dim, epi_dim=epi_dim, nhead=hla_nhead, dropout_rate=dropout_rate)
        self.epi_preprocessor = Epi_Preproc(hla_dim=hla_dim, epi_dim=epi_dim, nhead=epi_nhead, dropout_rate=dropout_rate)
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=epi_nhead, dropout=dropout_rate)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=epi_nhead, dropout=dropout_rate)

        concat_dim = 2*proj_dim
        nhead = self.find_optimal_nhead(concat_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=concat_dim, num_heads=nhead, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(concat_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(concat_dim, 1)

    def forward(self, x_hla, x_epi):
        query1, key1, value1 = self.hla_preprocessor(x_hla)
        query2, key2, value2 = self.epi_preprocessor(x_epi)

        # Align dimensions for cross-attention
        cross_out_1, _ = self.cross_attn_1(query1, key2, value2)
        cross_out_2, _ = self.cross_attn_2(query2, key1, value1)
        
        # Concatenate outputs and apply self-attention
        x = torch.cat((cross_out_1, cross_out_2), dim=-1)  # (1, batch_size, concat_dim)
        x, _ = self.self_attn(x, x, x)
        x = self.layer_norm(x)
        x = x.squeeze(0)  # (batch_size, concat_dim)
        x = self.dropout(x)
        x = torch.sigmoid(self.output_layer(x))
        return x
    
    def find_optimal_nhead(self, concat_dim):
        for nhead in range(10, 0, -1):
            if concat_dim % nhead == 0:
                return nhead
        return 1
    
#%%
class Alex_Basic(nn.Module):
    def __init__(self, hla_dim=1280, epi_dim=1280, num_heads=4, num_layers=2, hidden_dim=512):
        super(Alex_Basic, self).__init__()
        input_dim = hla_dim + epi_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 첫 번째 Linear Layer
            nn.BatchNorm1d(hidden_dim),         # Batch Normalization
            nn.ReLU(),                          # ReLU 활성화 함수
            nn.Dropout(0.5),                    # 50% 드롭아웃
    
            nn.Linear(hidden_dim, hidden_dim // 2), # 추가된 히든 레이어
            nn.BatchNorm1d(hidden_dim // 2),        # Batch Normalization
            nn.LeakyReLU(0.1),                      # LeakyReLU 활성화 함수
            nn.Dropout(0.5),                        # 50% 드롭아웃
    
            nn.Linear(hidden_dim // 2, 1),      # 출력 레이어
            nn.Sigmoid()                        # Sigmoid 활성화 함수
        )

    def forward(self, x_hla, x_epi):
        x_hla = x_hla.mean(dim=-2)              # HLA 시퀀스의 평균값
        x_epi = x_epi.mean(dim=-2)
        # print(x_hla.shape, x_epi.shape)
        x = torch.cat((x_hla, x_epi), dim=-1)   # HLA와 에피토프 시퀀스 연결
        x = self.transformer_encoder(x)         # 트랜스포머 인코더 통과
        x = self.classifier(x)                  # 분류기 통과
        return x
    
class cat2_alpha_sp(nn.Module):
    def __init__(
            self, 
            hla_dim_s=384, epi_dim_s=384, hla_dim_p=384, epi_dim_p=384, hla_nhead_s=8, 
            epi_nhead_s=5, hla_nhead_p=8, epi_nhead_p=5, d_model=128, dropout=0.2,
            hla_blocks=2, epi_blocks=2, con_blocks=2
        ):
        super(cat2_alpha_sp, self).__init__()
        self.hla_linear = nn.Linear(hla_dim_s+hla_dim_p, 512)
        self.epi_linear = nn.Linear(epi_dim_s+epi_dim_p, 512)
        
        self.epi_self_attn = simple_self_attn(embed_dim=512, num_heads=16, n_blocks=epi_blocks, dropout=dropout)
        self.hla_self_attn = simple_self_attn(embed_dim=512, num_heads=16, n_blocks=hla_blocks, dropout=dropout)
        
        concat_dim= 1024
        nhead = 16

        self.self_attn = simple_self_attn(embed_dim=1024, num_heads=nhead, n_blocks=con_blocks, dropout=dropout)
        self.output_layer = nn.Linear(concat_dim, 1)

    def forward(self, x_hla_s, x_hla_p, x_epi_s , x_epi_p):
        # HLA self-attention
        x_hla = torch.cat([x_hla_s, x_hla_p], dim=-1) # (batch, hla_len, emb_dim)
        x_hla = self.hla_linear(x_hla)
        x_hla = self.hla_self_attn(x_hla)
        x_hla = x_hla.mean(dim=1)
        x_hla = x_hla.unsqueeze(0)

        x_epi = torch.cat([x_epi_s, x_epi_p], dim=-1)   # (batch, epi_len, emb_dim)
        x_epi = self.epi_linear(x_epi)
        x_epi = self.epi_self_attn(x_epi)
        x_epi = x_epi.mean(dim=1)
        x_epi = x_epi.unsqueeze(0)
        
        x = torch.cat((x_hla, x_epi), dim=-1)    # (1, batch_size, concat_dim)
        x = self.self_attn(x)
        x = x.squeeze(0)

        x = self.output_layer(x)
        return x

class IM_alpha_sp(nn.Module):
    def __init__(
            self, 
            hla_dim_s=384, epi_dim_s=384, hla_dim_p=384, epi_dim_p=384, hla_nhead_s=8, dim_bind=1074,
            epi_nhead_s=5, hla_nhead_p=8, epi_nhead_p=5, d_model=128, dropout=0.2,
            hla_blocks=2, epi_blocks=2, con_blocks=2
        ):
        super(IM_alpha_sp, self).__init__()
        self.hla_linear = nn.Linear(hla_dim_s+hla_dim_p, 512)
        self.epi_linear = nn.Linear(epi_dim_s+epi_dim_p, 512)
        
        self.epi_self_attn = simple_self_attn(embed_dim=512, num_heads=16, n_blocks=epi_blocks, dropout=dropout)
        self.hla_self_attn = simple_self_attn(embed_dim=512, num_heads=16, n_blocks=hla_blocks, dropout=dropout)
        
        concat_dim= 1024 + dim_bind
        nhead = 16

        self.self_attn = simple_self_attn(embed_dim=1024, num_heads=nhead, n_blocks=con_blocks, dropout=dropout)
        self.output_layer = nn.Linear(concat_dim, 1)

    def forward(self, x_hla_s, x_hla_p, x_epi_s , x_epi_p, emb_bind):
        # HLA self-attention
        x_hla = torch.cat([x_hla_s, x_hla_p], dim=-1) # (batch, hla_len, emb_dim)
        x_hla = self.hla_linear(x_hla)
        x_hla = self.hla_self_attn(x_hla)
        x_hla = x_hla.mean(dim=1)
        x_hla = x_hla.unsqueeze(0)

        x_epi = torch.cat([x_epi_s, x_epi_p], dim=-1)   # (batch, epi_len, emb_dim)
        x_epi = self.epi_linear(x_epi)
        x_epi = self.epi_self_attn(x_epi)
        x_epi = x_epi.mean(dim=1)
        x_epi = x_epi.unsqueeze(0)
        
        x = torch.cat((x_hla, x_epi, emb_bind), dim=-1)    # (1, batch_size, concat_dim)
        x = self.self_attn(x)
        x = x.squeeze(0)

        x = self.output_layer(x)
        return x
