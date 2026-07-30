# 红星杯项目 — Red Star Project

## 一句话目标

摄像头看到中国手语（CSL），实时输出中文文本。

## 核心思路

```
摄像头 → MediaPipe 手部关键点 → 时序模型预测 → 中文词序列
```

传统方案用 ResNet 逐帧提取特征（21M 参数，慢），本项目用 MediaPipe 提取手部 42 个关键点坐标来代替，模型轻量、支持实时推理。当前模型：1D Conv(stride=4) + 4 层 Conformer Encoder + CTC 解码。

## 项目结构速览

```
项目根目录/
├── docs/
│   ├── permanent/           # 永久文档
│   │   ├── ARCHITECTURE.md  # 架构设计（权威文档，新同学先读这个）
│   │   ├── IMPLEMENTATION.md# 实现计划和工作量估算
│   │   ├── CHANGELOG.md     # 工作日志
│   │   └── INSPIRATION.md   # 启发日志
│   └── temporary/           # 临时实验文档
│
├── TFNet-main/            # 原论文代码（仅复用 WER.py）
├── CE-CSL/CE-CSL/         # 中国手语数据集（6000条句子视频）
├── CSL_basic_dataset/     # 手语基础词（235个词视频）
├── CSL_common_dataset/    # 手语常用词（863个词视频）
├── YOLOv8/                # YOLO 检测模块
│
├── src/                         # 项目源代码（所有核心模块）
│   ├── build_vocab.py           # 词表构建
│   ├── preprocess_keypoints.py  # 预处理（视频→关键点）
│   ├── model.py                 # 1D Conv(stride=4) + Conformer + Linear
│   ├── dataset.py               # KeypointDataset + collate_fn
│   ├── train.py                 # 训练脚本（CTC + blank penalty + 熵正则）
│   └── decode.py                # CTC 贪心解码
├── models/                      # 模型文件
├── checkpoints/                 # 训练权重
├── vocab.json                   # 词表（build_vocab.py 生成，3515 tokens）
└── requirements.txt             # Python 依赖
```

## 当前进度（2026-07-30）

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 环境搭建 | 完成 |
| Phase 1 | 关键点预处理 | 完成（5987 .npy） |
| Phase 2 | 主路模型训练 | baseline WER 66.85%（Conformer+activity detection） |
| Phase 3 | CTC 解码集成 | 代码完成，blank penalty + 熵正则已集成 |
| Phase 4 | 实时推理管线 | 待开始 |
| Phase 5 | 旁路 YOLO 分类 | 待开始 |
| Phase 6 | 系统联调测试 | 待开始 |

### 视觉特征融合实验（2026-07-30）

用 MobileNetV3-Small 提取的 576d 手部视觉特征与 84d 关键点融合：

| 实验 | WER | 结论 |
|------|-----|------|
| kp-only baseline（全量） | 66.58% | 当前最优 |
| kp-only（子集 4972） | 80.94% | 子集本身更难 |
| Raw concat visual | 84.79% | collapse=0%，但噪声淹没了信号 |
| Projected fusion | 91.61% | 噪声获得不对等容量，更差 |

**结论**：ImageNet 预训练特征编码的是物体类别，与手语无关。视觉融合路线暂搁置，v2 考虑 domain-specific 手部视觉 encoder。

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

# 训练
python src/build_vocab.py        # 构建词表（已生成 vocab.json）
python src/train.py              # 训练模型

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
