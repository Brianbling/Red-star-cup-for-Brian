"""
主路时序模型：1D Conv (stride=4) + Conformer Encoder + Linear + LogSoftmax
+ 孤立词分类组件（ConformerEncoder + ClassificationHead）

visual_fusion 模式：
- "raw": 直接 concat kp(84d) + vis(576d) -> 660d
- "projected": 分别投影 kp(84->128) + vis(576->128) -> 256d
- "none": 仅 kp(84d)
"""
import math
import torch
import torch.nn as nn


# ─── Conformer building blocks (from current training architecture) ──────────

class Conv2dSubsampling(nn.Module):
    """Stride=4 降采样: (T, B, D) -> (T/4, B, d_model)"""
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, input_lengths):
        """x: (T, B, D) time-major"""
        x = x.permute(1, 2, 0)  # (B, D, T)
        x = self.conv(x)
        x = x.permute(2, 0, 1)  # (T', B, d_model)
        new_lengths = (input_lengths // 4).clamp(min=1)
        return x, new_lengths


class ConformerEncoderBlock(nn.Module):
    """Single Conformer block: MHSA + Conv + FFN, pre-norm style."""
    def __init__(self, d_model, num_heads, ff_dim=1024, conv_kernel=15, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                               batch_first=False)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, conv_kernel, padding=conv_kernel // 2,
                      groups=d_model),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, 1),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm_mhsa = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (T, B, d_model) time-major"""
        # Self-attention with residual
        residual = x
        x = self.norm_mhsa(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(attn_out)

        # Conv module
        residual = x
        x = self.norm_conv(x)
        x_conv = x.permute(1, 2, 0)  # (B, D, T)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.permute(2, 0, 1)  # (T, B, D)
        x = residual + self.dropout(x_conv)

        # FFN
        residual = x
        x = self.norm_ffn(x)
        x = residual + self.dropout(self.ffn(x))
        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder: Conv2dSubsampling + N x ConformerEncoderBlock."""
    def __init__(self, input_dim=84, d_model=256, num_heads=4, num_blocks=4,
                 ff_dim=1024, conv_kernel=15, dropout=0.3):
        super().__init__()
        self.subsampling = Conv2dSubsampling(input_dim, d_model)
        self.blocks = nn.ModuleList([
            ConformerEncoderBlock(d_model, num_heads, ff_dim, conv_kernel, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x, input_lengths):
        """x: (T, B, input_dim) time-major"""
        x, lengths = self.subsampling(x, input_lengths)
        for block in self.blocks:
            x = block(x)
        return x, lengths


# ─── Classification head for isolated word pretraining ──────────────────────

class ClassificationHead(nn.Module):
    """Attention or mean pooling + Linear -> class logits."""
    def __init__(self, d_model, num_classes, pooling="attention", dropout=0.3):
        super().__init__()
        self.pooling = pooling
        if pooling == "attention":
            self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, encoder_output, input_lengths=None):
        """encoder_output: (T, B, d_model) time-major"""
        if self.pooling == "attention":
            T, B, D = encoder_output.shape
            x = encoder_output.permute(1, 0, 2)  # (B, T, D)
            scores = torch.matmul(x, self.query) / math.sqrt(D)
            attn = torch.softmax(scores, dim=-1).unsqueeze(-1)
            pooled = (x * attn).sum(dim=1)  # (B, D)
        else:
            pooled = encoder_output.mean(dim=0)  # (B, D)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


# ─── Full CTC model (for continuous sentence training) ──────────────────────

class SLRModel(nn.Module):
    def __init__(self, vocab_size, input_dim=660, d_model=256, num_heads=4,
                 num_blocks=4, dropout=0.3, visual_fusion="raw",
                 kp_dim=84, vis_dim=576, fusion_dim=128):
        super().__init__()
        self.visual_fusion = visual_fusion

        if visual_fusion == "projected":
            self.kp_proj = nn.Linear(kp_dim, fusion_dim)
            self.vis_proj = nn.Linear(vis_dim, fusion_dim)
            effective_input_dim = fusion_dim * 2
        else:
            effective_input_dim = input_dim

        self.encoder = ConformerEncoder(
            input_dim=effective_input_dim, d_model=d_model,
            num_heads=num_heads, num_blocks=num_blocks, dropout=dropout,
        )
        self.fc = nn.Linear(d_model, vocab_size)
        nn.init.constant_(self.fc.bias, 0.0)
        self.fc.bias.data[0] = 5.32
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        if self.visual_fusion == "projected":
            kp = x[:, :, :84]
            vis = x[:, :, 84:]
            kp = self.kp_proj(kp)
            vis = self.vis_proj(vis)
            x = torch.cat([kp, vis], dim=-1)

        x, lengths = self.encoder(x, input_lengths)
        logits = self.fc(x)
        return self.log_softmax(logits)
