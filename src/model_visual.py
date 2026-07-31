"""
视觉端到端模型：MobileNetV3Backbone（逐帧 128d）→ Conv(stride=4) → BiLSTM → FC + LogSoftmax。

与 SLRModel 的后段完全一致，仅把输入从关键点换成 backbone 特征。
输入:  (T, B, 3, 224, 224)  原始视频帧
输出:  (T/4, B, vocab_size) log 概率

逐帧处理：backbone 每次吃一批帧 (chunk, 3, 224, 224)，在帧维循环，控制显存。
"""
import torch
import torch.nn as nn
from visual_backbone import MobileNetV3Backbone


class VisualSLRModel(nn.Module):
    def __init__(self, vocab_size, backbone_out_dim=128, conv_dim=256,
                 hidden_size=512, num_layers=2, dropout=0.3, blank_bias=None,
                 backbone_chunk=30):
        super().__init__()
        self.backbone = MobileNetV3Backbone(out_dim=backbone_out_dim)
        self.backbone_chunk = backbone_chunk

        self.conv = nn.Sequential(
            nn.Conv1d(backbone_out_dim, conv_dim, kernel_size=3, padding=1),
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
        if blank_bias is not None:
            self.fc.bias.data[0] = blank_bias
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def extract_frame_features(self, frames):
        """frames: (T, B, 3, 224, 224) → (T, B, backbone_out_dim)

        逐帧循环，每块 backbone_chunk 帧一次 forward。
        """
        T, B = frames.shape[0], frames.shape[1]
        outs = []
        for start in range(0, T, self.backbone_chunk):
            chunk = frames[start:start + self.backbone_chunk]  # (c, B, 3, 224, 224)
            c = chunk.shape[0]
            flat = chunk.reshape(c * B, 3, 224, 224)
            feats = self.backbone(flat)  # (c*B, 128)
            feats = feats.reshape(c, B, -1)
            outs.append(feats)
        return torch.cat(outs, dim=0)  # (T, B, 128)

    def forward(self, x, input_lengths):
        # x: (T, B, 3, 224, 224)
        feats = self.extract_frame_features(x)  # (T, B, 128)

        # feats: (T, B, D) → conv1d 需要 (B, D, T)
        conv_in = feats.permute(1, 2, 0)
        conv_out = self.conv(conv_in)  # (B, 256, T/4)
        conv_out = conv_out.permute(0, 2, 1)  # (B, T/4, 256)
        conv_out = self.layer_norm(conv_out)  # (B, T/4, 256)
        lstm_in = conv_out.permute(1, 0, 2)  # (T/4, B, 256)

        input_lengths = (input_lengths // 4).clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(lstm_in, input_lengths.cpu(),
                                                   enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out)

        logits = self.fc(lstm_out)
        return self.log_softmax(logits)
