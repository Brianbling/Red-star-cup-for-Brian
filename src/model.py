"""
主路时序模型：1D Conv (stride=2×2 降采样) + BiLSTM + Linear + LogSoftmax
输入 (T, B, input_dim) 多模态序列，输出 (T/4, B, vocab_size) log 概率。
"""
import torch
import torch.nn as nn


class SLRModel(nn.Module):
    def __init__(self, vocab_size, input_dim=660, conv_dim=256, hidden_size=512,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=3, padding=1),
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
        self.fc.bias.data[0] = 5.32  # P(blank) = σ(5.32) ≈ 0.3
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        x = x.permute(1, 2, 0)  # (B, input_dim, T)
        x = self.conv(x)         # (B, conv_dim, T/4)
        x = x.permute(0, 2, 1)  # (B, T', conv_dim)
        x = self.layer_norm(x)
        x = x.permute(1, 0, 2)  # (T', B, conv_dim)

        input_lengths = (input_lengths // 4).clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(),
                                                    enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out)

        logits = self.fc(lstm_out)
        return self.log_softmax(logits)
