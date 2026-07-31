# 红星光项目 — Red Star Project

## 一句话目标

摄像头看到中国手语（CSL），实时输出中文文本。

## 核心思路

```
摄像头 → MediaPipe 手部关键点 → 时序模型预测 → 中文词序列
```

传统方案用 ResNet 逐帧提取特征（21M 参数，慢），本项目用 MediaPipe 提取手部 42 个关键点坐标来代替，模型轻量、支持实时推理。当前模型：1D Conv（总降采样 /4）+ BiLSTM(2层,双向,hidden=512) + CTC 解码。

## 项目结构速览

```
项目根目录/
├── docs/
│   ├── permanent/           # 永久文档
│   │   ├── ARCHITECTURE.md  # 架构设计（权威文档，新同学先读这个）
│   │   ├── IMPLEMENTATION.md# 实现计划和工作量估算
│   │   ├── CHANGELOG.md     # 工作日志
│   │   └── INSPIRATION.md   # 启发日志
│   └── temporary/           # 临时实验文档（实验方案.md 未入库）
│
├── TFNet-main/            # 原论文代码（仅复用 WER.py）
├── CE-CSL/CE-CSL/         # 中国手语数据集（6000条句子视频）
├── CSL_basic_dataset/     # 手语基础词（235个词视频）
├── CSL_common_dataset/    # 手语常用词（863个词视频）
├── L1290/                 # C382 手势手语数据（35类，YOLO 检测格式）
├── SLR_Dataset/           # CSL-2015（25K 孤立词 + 100 句连续）
├── isolated_words/        # 孤立词关键点缓存（1098 .npy + index.json）
├── YOLOv8/                # YOLO 检测模块（L1290 权重在 runs/l1290/）
│
├── src/                         # 项目源代码（所有核心模块）
│   ├── build_vocab.py           # 词表构建
│   ├── preprocess_keypoints.py  # 预处理（视频→关键点，含 normalize_hand）
│   ├── model.py                 # 1D Conv(/4) + BiLSTM(2层,双向) + Linear
│   ├── dataset.py               # KeypointDataset + collate_fn
│   ├── train.py                 # 训练脚本（CTC + blank penalty + 熵正则）
│   ├── decode.py                # CTC 贪心解码
│   ├── activity_detect.py       # 切手部丢失帧
│   ├── vocab_utils.py           # clean_word() 标签清洗
│   └── ...                      # 实验脚本见 CLAUDE.md 目录结构
├── models/                      # 模型文件
├── checkpoints/                 # 训练权重
├── vocab.json                   # 词表（build_vocab.py 生成，3515 tokens）
└── requirements.txt             # Python 依赖
```

## 当前进度（2026-08-01）

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 环境搭建 | 完成 |
| Phase 1 | 关键点预处理 | 完成（5987 .npy，含归一化） |
| Phase 2 | 主路模型训练 | 实验完成（WER 66.58%），~5K 数据是唯一瓶颈 |
| Phase 3 | CTC 解码集成 | 完成，贪心解码 + blank penalty + 熵正则（beam 验证等价） |
| Phase 4 | 实时推理管线 | 待开始 |
| Phase 5 | 旁路 YOLO 分类 | 分类模型已训好（L1290 mAP50=0.985），集成待做 |
| Phase 6 | 系统联调测试 | 待开始 |

### 结论速览（2026-07-30 → 2026-08-01）

- **WER 瓶颈 = 数据量（~5K 连续句样本）**，不是模型架构（BiLSTM 13.5M）、特征维度（84d）、解码策略、归一化或标签完整性。已排除 blank 坍塌、视觉特征融合、activity detection、分组 padding、时序增强、beam search
- **视觉特征融合失败**：MobileNetV3 ImageNet 特征与手语无关，raw concat 84.79%、projected 91.61%（vs kp-only 66.58%）
- **零成本优化三项均退步或不改善**：分组 padding 68.83%、时序增强 75.69%、beam search 0pp
- **L1290 旁路分类已训好**：mAP50=0.985（epoch 18）
- **孤立词数据扩展已交付**：1098 .npy + index.json（87 token / 94 匹配视频），CombinedDataset 可混入 CE-CSL train
- **下一步**：SLR 25K 孤立词预训练 + CSL-Daily 20K 数据扩展（`docs/temporary/实验方案.md`）

## 环境

```bash
pip install -r requirements.txt
```

| 依赖 | 版本 |
|------|------|
| PyTorch | 2.7.1+cu128 |
| MediaPipe | 0.10.35 |
| OpenCV | 5.0.0 |
| numpy | 2.2.6 |
| ultralytics | 8.4.105 |

## 怎么跑

```bash
# 预处理：视频 → 关键点（先小批量验证）
python src/preprocess_keypoints.py --split train --max_videos 10

# 训练（默认 kp-only 84d 输入）
python src/build_vocab.py        # 构建词表（已生成 vocab.json）
python src/train.py              # 训练模型（batch_size=2，blank_bias=+5.32）

# 旁路 YOLO 训练（L1290 已完成，产物在 YOLOv8/runs/l1290/）
python YOLOv8/train_l1290.py

# 更多操作见 .claude/commands/run.md，或问算法同学
```

## 给前端同学

算法最终输出的格式：**每帧返回的中文字符串**。你需要做的：

- 一个摄像头预览画面 + 叠加识别文本的 UI
- 按 Q 退出
- （可选）显示当前识别状态：动态/静默

算法侧暂定通过 Python API 调用，后续可以改成 HTTP/WebSocket。

## 给后端同学

暂时不涉及后端。后续如果需要模型云端部署，模型格式：PyTorch `.pt`，输入 `(T, 84)` 关键点序列，输出中文词序列。
