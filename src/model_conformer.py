"""
分支 B — Conformer Encoder（无 BiLSTM）
Conv2dSubsampling(stride=4) + PositionalEncoding + 4×ConformerBlock + Linear → vocab
轻量配置: d_model=256, heads=4, conv_kernel=15, ~8-10M 参数
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSubsampling(nn.Module):
    def __init__(self, input_dim, d_model, stride=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, T, 84) → (B, 84, T)
        x = self.conv(x)          # (B, d_model, T/4)
        x = x.permute(0, 2, 1)   # (B, T/4, d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, conv_kernel_size, ff_dim_ratio=4, dropout=0.1):
        super().__init__()
        self.ff_scale = 0.5  # macaron style

        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_dim_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_dim_ratio, d_model),
            nn.Dropout(dropout),
        )

        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mhsa_dropout = nn.Dropout(dropout)

        self.conv_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(d_model, d_model, conv_kernel_size, padding=conv_kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_dropout = nn.Dropout(dropout)

        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_dim_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_dim_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.ff_scale * self.ffn1(x)

        residual = x
        x = self.mhsa_norm(x)
        x = residual + self.mhsa_dropout(self.mhsa(x, x, x, need_weights=False)[0])

        residual = x
        x = self.conv_norm(x)
        x = x.permute(0, 2, 1)              # (B, C, T)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pointwise2(x)
        x = x.permute(0, 2, 1)              # (B, T, C)
        x = residual + self.conv_dropout(x)

        x = x + self.ff_scale * self.ffn2(x)

        return x


class Conformer(nn.Module):
    def __init__(self, vocab_size, input_dim=84, d_model=256, num_blocks=4,
                 num_heads=4, conv_kernel_size=15, dropout=0.1, blank_id=0):
        super().__init__()
        self.subsampling = Conv2dSubsampling(input_dim, d_model)
        self.pe = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel_size, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        nn.init.constant_(self.fc.bias, 0.0)
        self.fc.bias.data[blank_id] = 5.32
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        x = x.permute(1, 0, 2)  # (T, B, 84) → (B, T, 84)
        x = self.subsampling(x)  # (B, T/4, d_model)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.fc(x)      # (B, T/4, vocab)
        logits = logits.permute(1, 0, 2)  # (T/4, B, vocab)
        return self.log_softmax(logits)
