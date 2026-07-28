"""
主路时序模型: 1D Conv + BiLSTM + Linear + LogSoftmax
输入 (T, B, 84) 关键点序列，输出 (T, B, vocab_size) log 概率。
"""
import torch
import torch.nn as nn


class SLRModel(nn.Module):
    def __init__(self, vocab_size, input_dim=84, conv_dim=256, hidden_size=512,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(conv_dim),
        )
        self.lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        """
        x: (T, B, input_dim)  time-major
        input_lengths: (B,) int64 tensor, 每样本实际帧数
        返回: (T, B, vocab_size) log probabilities
        """
        # Conv1d 需要 (B, C, T)
        x = x.permute(1, 2, 0)  # (B, input_dim, T)
        x = self.conv(x)         # (B, conv_dim, T)
        x = x.permute(2, 0, 1)  # (T, B, conv_dim)

        # pack → LSTM → unpack
        packed = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(),
                                                    enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out)  # (T, B, hidden*2)

        logits = self.fc(lstm_out)  # (T, B, vocab_size)
        return self.log_softmax(logits)
