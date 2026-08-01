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


class SequenceVAE(nn.Module):
    """轻量 VAE 辅助头：逐帧对 BiLSTM 序列特征做重建正则（β-VAE 风格）。

    训练时辅助 CTC，推理时不参与，零额外开销。
    """
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std
        recon = self.decoder(z)
        return recon, mu, log_var


class SLRModel(nn.Module):
    def __init__(self, vocab_size, input_dim=660, conv_dim=256, hidden_size=512,
                 num_layers=2, dropout=0.3, visual_fusion="raw",
                 kp_dim=84, vis_dim=576, fusion_dim=128, blank_bias=5.32,
                 vae=False, vae_latent_dim=64):
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

        self.vae = SequenceVAE(hidden_size * 2, latent_dim=vae_latent_dim) if vae else None

    def forward(self, x, input_lengths, return_features=False, return_vae_loss=False):
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
        log_probs = self.log_softmax(logits)

        vae_loss = None
        if return_vae_loss and self.vae is not None:
            recon, mu, log_var = self.vae(lstm_out)
            valid_mask = torch.arange(lstm_out.shape[0], device=lstm_out.device).unsqueeze(1) < input_lengths.unsqueeze(0)
            recon_err = ((recon - lstm_out) ** 2).sum(dim=-1)[valid_mask].mean()
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())[valid_mask].sum(dim=-1).mean()
            vae_loss = recon_err + kl

        if return_features:
            return log_probs, lstm_out
        if return_vae_loss:
            return log_probs, vae_loss
        return log_probs
