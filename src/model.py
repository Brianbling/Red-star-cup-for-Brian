"""
主路时序模型：1D Conv (stride=2×2 降采样) + BiLSTM + Linear + LogSoftmax
输入 (T, B, input_dim) 多模态序列，输出 (T/4, B, vocab_size) log 概率。

visual_fusion 模式：
- "raw": 直接 concat kp(84d) + vis(576d) → 660d（早期融合）
- "projected": 分别投影 kp(84→128) + vis(576→128) → 256d（等维度融合）
- "none": 仅 kp(84d)，无视觉特征
"""
import torch
import torch.nn as nn


class SLRModel(nn.Module):
    def __init__(self, vocab_size, input_dim=660, conv_dim=256, hidden_size=512,
                 num_layers=2, dropout=0.3, visual_fusion="raw",
                 kp_dim=84, vis_dim=576, fusion_dim=128, blank_bias=5.32):
        super().__init__()
        self.visual_fusion = visual_fusion

        if visual_fusion == "projected":
            self.kp_proj = nn.Linear(kp_dim, fusion_dim)
            self.vis_proj = nn.Linear(vis_dim, fusion_dim)
            effective_input_dim = fusion_dim * 2
        else:
            effective_input_dim = input_dim

        self.conv = nn.Sequential(
            nn.Conv1d(effective_input_dim, conv_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer_norm = nn.LayerNorm(conv_dim)
        self.lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        nn.init.constant_(self.fc.bias, 0.0)
        self.fc.bias.data[0] = blank_bias
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        if self.visual_fusion == "projected":
            kp = x[:, :, :84]
            vis = x[:, :, 84:]
            kp = self.kp_proj(kp)
            vis = self.vis_proj(vis)
            x = torch.cat([kp, vis], dim=-1)

        x = x.permute(1, 2, 0)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(1, 0, 2)

        input_lengths = (input_lengths // 4).clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(),
                                                    enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out)

        logits = self.fc(lstm_out)
        return self.log_softmax(logits)
