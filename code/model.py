import torch.nn.functional as F
import torch.nn as nn
import torch

#%% DEEPNEO
class DeepNeo(nn.Module):
    def __init__(self, kernel_size=(8, 133)):
        super(DeepNeo, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=kernel_size, stride=1) # HLA1: (5, 183), HLA2: (8, 133)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=10, kernel_size=kernel_size, stride=1)
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

#%% PLM BLOCKS
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

    def forward(self, x, padding_mask=None):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=padding_mask)
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
        
    def forward(self, x, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask)
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

#%% PLM MODELS
class plm_cat_flat(nn.Module):
    def __init__(
            self, 
            hla_dim_s=384, epi_dim_s=384, hla_dim_p=384, epi_dim_p=384, 
            dropout=0.2, hla_blocks=2, epi_blocks=2, con_blocks=2, head_div=64
        ):
        super(plm_cat_flat, self).__init__()
        self.pair=False if hla_dim_p == 0 and epi_dim_p == 0 else True
        
        hla_dim = hla_dim_s + hla_dim_p # 640
        epi_dim = epi_dim_s + epi_dim_p # 640
        nhead = hla_dim // head_div
        self.epi_self_attn = simple_self_attn(embed_dim=hla_dim, num_heads=nhead, n_blocks=epi_blocks, dropout=dropout)
        self.hla_self_attn = simple_self_attn(embed_dim=epi_dim, num_heads=nhead, n_blocks=hla_blocks, dropout=dropout)
        
        concat_dim = hla_dim
        nhead = concat_dim // head_div
        self.self_attn = simple_self_attn(embed_dim=concat_dim, num_heads=nhead, n_blocks=con_blocks, dropout=dropout)
        
        flat_dim = concat_dim*(15+269)
        self.output_layer = nn.Sequential(
            nn.Linear(flat_dim, flat_dim//2),
            nn.ReLU(),
            nn.Linear(flat_dim//2, 1)
        )

    def forward(self, x_hla_s, x_hla_p, x_epi_s, x_epi_p, mask_hla=None, mask_epi=None):
        x_hla = torch.cat([x_hla_s, x_hla_p], dim=-1) if self.pair else x_hla_s
        x_hla = self.hla_self_attn(x_hla, padding_mask=mask_hla)

        x_epi = torch.cat([x_epi_s, x_epi_p], dim=-1) if self.pair else x_epi_s
        x_epi = self.epi_self_attn(x_epi, padding_mask=None)
        
        mask = torch.cat((mask_hla, mask_epi), dim=-1)
        x = torch.cat((x_hla, x_epi), dim=1)
        x = self.self_attn(x, padding_mask=mask)
        
        x = torch.flatten(x, start_dim=1)
        x = self.output_layer(x)
        return x

class plm_cat_mean(nn.Module):
    def __init__(
            self, 
            hla_dim_s=384, epi_dim_s=384, hla_dim_p=384, epi_dim_p=384, 
            dropout=0.2, hla_blocks=2, epi_blocks=2, con_blocks=2, head_div=64
        ):
        super(plm_cat_mean, self).__init__()
        self.pair=False if hla_dim_p == 0 and epi_dim_p == 0 else True

        hla_dim = hla_dim_s + hla_dim_p # 640
        epi_dim = epi_dim_s + epi_dim_p # 640
        nhead = hla_dim // head_div
        self.epi_self_attn = simple_self_attn(embed_dim=hla_dim, num_heads=nhead, n_blocks=epi_blocks, dropout=dropout)
        self.hla_self_attn = simple_self_attn(embed_dim=epi_dim, num_heads=nhead, n_blocks=hla_blocks, dropout=dropout)
        
        concat_dim = hla_dim
        nhead = concat_dim // head_div
        self.self_attn = simple_self_attn(embed_dim=concat_dim, num_heads=nhead, n_blocks=con_blocks, dropout=dropout)
        
        flat_dim = concat_dim
        self.output_layer = nn.Sequential(
            nn.Linear(flat_dim, flat_dim//2),
            nn.ReLU(),
            nn.Linear(flat_dim//2, 1)
        )

    def forward(self, x_hla_s, x_hla_p, x_epi_s, x_epi_p, mask_hla=None, mask_epi=None):
        x_hla = torch.cat([x_hla_s, x_hla_p], dim=-1) if self.pair else x_hla_s
        x_hla = self.hla_self_attn(x_hla, padding_mask=mask_hla)

        x_epi = torch.cat([x_epi_s, x_epi_p], dim=-1) if self.pair else x_epi_s
        x_epi = self.epi_self_attn(x_epi, padding_mask=None)
        
        mask = torch.cat((mask_hla, mask_epi), dim=-1)
        x = torch.cat((x_hla, x_epi), dim=1)
        x = self.self_attn(x, padding_mask=mask)
        
        mask = 1-mask.unsqueeze(-1)
        x = x*mask
        x = x.sum(dim=1) / (mask.sum(dim=1))
        x = self.output_layer(x)
        return x