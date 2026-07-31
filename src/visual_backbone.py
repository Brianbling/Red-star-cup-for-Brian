"""
MobileNetV3-Large 视觉骨干（端到端 fine-tune）。
输入 (B, 3, 224, 224) → 输出 (B, 128)。

freeze_stages: 冻结前 N 个 stage。每个 stage 结束于一个 stride=2 的 InvertedResidual。
stage 划分（按 mobileNetV3 官方 block 结构）：
  stage0: [0]                       stem (stride 2)
  stage1: [1,2,3]                   (stride 2 at block 2)
  stage2: [4,5,6]                   (stride 2 at block 4)
  stage3: [7,8,9,10,11,12]          (stride 2 at block 7)
  stage4: [13,14,15,16]             (stride 2 at block 13)

freeze_stages=3 → 冻结 blocks 0..12，微调 blocks 13..16 + classifier。
"""
import torch.nn as nn
from torchvision import models

STAGES = [
    [0],
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16],
]


class MobileNetV3Backbone(nn.Module):
    def __init__(self, out_dim=128, freeze_stages=3):
        super().__init__()
        self.freeze_stages = freeze_stages
        self.mobilenet = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        )

        frozen_blocks = set()
        for stage_idx in range(freeze_stages):
            frozen_blocks.update(STAGES[stage_idx])

        for i, block in enumerate(self.mobilenet.features):
            requires_grad = i not in frozen_blocks
            for p in block.parameters():
                p.requires_grad = requires_grad

        # 替换分类头
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x
