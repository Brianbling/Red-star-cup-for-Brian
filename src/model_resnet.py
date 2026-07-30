"""
分支 A — Conv+ResNet 模型（无 BiLSTM）
stride=4 前置卷积 + 3 个 ResBlock + Linear → vocab
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ConvResNet(nn.Module):
    def __init__(self, vocab_size, input_dim=84, conv_dim=256, num_blocks=6,
                 blank_id=0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(conv_dim) for _ in range(num_blocks)])
        self.fc = nn.Linear(conv_dim, vocab_size)
        nn.init.constant_(self.fc.bias, 0.0)
        self.fc.bias.data[blank_id] = 5.32
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        x = x.permute(1, 2, 0)  # (T, B, 84) → (B, 84, T)
        x = self.stem(x)         # (B, 256, T/4)
        x = self.blocks(x)       # (B, 256, T/4)
        x = x.permute(2, 0, 1)   # (T/4, B, 256)
        logits = self.fc(x)      # (T/4, B, vocab)
        return self.log_softmax(logits)
