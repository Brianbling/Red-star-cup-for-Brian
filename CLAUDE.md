# 红星杯项目 (Red Star Project)

## 目标

中国手语（CSL）实时连续识别。摄像头视频流输入 → 手语文本实时输出。

## 系统架构

详见 `ARCHITECTURE.md`（权威）。核心管线：

```
摄像头 → 统一预处理 → MediaPipe Hands (42点关键点)
                      │
                      ├─ 主路: 置信度清洗 → 坐标归一化 → 滑动窗口(3s)
                      │        → 1D Conv + BiLSTM → CTC解码 → 后处理去重 → 文本
                      │
                      └─ 旁路: YOLO 静态手势分类(每5帧) → 三重AND门控 → 静态词
                               ↑ 可关闭，低算力设备退化为纯时序识别
```

融合仲裁层：速度门控 + 防抖计时器 + 三态状态机（动态/静默/缓冲）。

## 当前进度

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 环境搭建 | 已完成 (2026-07-28) |
| 1 | 关键点预处理（5987 .npy，~0.4GB） | **已完成** (2026-07-29) |
| 2 | 模型训练 | 代码完成，待启动 |
| 3 | CTC 解码 | 代码完成 (`src/decode.py`) |
| 4 | 实时推理管线 | 待开始 |
| 5 | 旁路 YOLO | 待开始 |
| 6 | 联调测试 | 待开始 |

主路训练计划：先 CE-CSL 6000 句跑 baseline WER → CSL-Daily（下载中）单独训对比 → 合并训最终模型。

## 项目结构

```
D:/red star project/
├── ARCHITECTURE.md           # 架构设计（新同学先读这个）
├── CLAUDE.md                 # 本文件：入口指南
├── CHANGELOG.md              # 工作日志 + 决策记录
├── IMPLEMENTATION.md         # 实现计划（预估，代码为准）
├── README.md                 # 项目概览
├── requirements.txt          # Python 依赖
├── vocab.json                # 词表（3515 tokens）
│
├── src/                      # 项目源代码（所有核心模块）
│   ├── preprocess_keypoints.py # Phase 1: 视频 → 关键点 .npy（多进程）
│   ├── build_vocab.py        # 词表构建
│   ├── model.py              # 1D Conv + BiLSTM + Linear（13.1M 参数）
│   ├── dataset.py            # KeypointDataset + collate_fn
│   ├── train.py              # CTC Loss 训练脚本
│   └── decode.py             # CTC 贪心解码 + 后处理
│
├── CE-CSL/CE-CSL/            # 主路训练数据
│   ├── video/{train,dev,test}/  # 6000 条 .mp4
│   ├── label/{train,dev,test}.csv
│   └── keypoints/{train,dev,test}/  # 预处理缓存（5987 .npy，~0.4GB）
│
├── CSL_basic_dataset/        # 旁路：235 个词视频（无 CSV，文件名即标签）
├── CSL_common_dataset/       # 旁路：863 个词视频（无 CSV，文件名即标签）
├── SLR_Dataset/              # CSL-2015：25K 孤立词 + 100 句连续
├── ASL Alphabet/             # 美式手语字母（旁路预训练参考）
│
├── TFNet-main/               # 原论文代码，仅复用 WER.py
├── YOLOv8/                   # YOLO 检测（已有 MNIST demo）
├── ctc_decoders-master/      # C++ CTC 解码库（未编译/未使用）
├── reference/                # 外部参考项目（只读）
│   ├── wenet-main/           #   流式 ASR 生产框架
│   ├── espnet-master/        #   流式 ASR 研究框架
│   ├── landmark-extraction-main/  # 手部关键点预处理工具链
│   ├── slt-master/           #   手语翻译模型
│   └── k2-master/            #   FSA/FST 框架
│
├── models/hand_landmarker.task  # MediaPipe 模型 (~7.6MB)
├── checkpoints/              # 训练权重保存目录
└── .claude/                  # Claude Code 配置
    └── INSPIRATION.md        # 外部启发（待讨论，未采纳）
```

### 外部数据集（跨盘）

| 数据集 | 路径 | 规模 | 用途 | 状态 |
|--------|------|------|------|------|
| CSL-Daily | `E:/CSL-Daily.rar` | ~20,654 句 | 主路 CTC | 下载中 |

## 技术栈

| 项 | 版本 | 状态 |
|----|------|------|
| Python | — | ✓ |
| PyTorch | 2.7.1+cu128 | ✓ GPU 可用 |
| MediaPipe | 0.10.35 (tasks API, IMAGE 模式) | ✓ |
| OpenCV | 5.0.0 | ✓ |
| numpy | 2.2.6 | ✓ |
| ultralytics | 8.4.105 | ✓ |

## 快速开始

```bash
# 预处理：视频 → 关键点（多进程，断点续跑）
python src/preprocess_keypoints.py --split all          # 全量，所有 CPU 核心
python src/preprocess_keypoints.py --split train --max_videos 10 --workers 2  # 验证

# 训练
python src/build_vocab.py     # 构建词表（已生成 vocab.json，3515 tokens）
python src/train.py           # CTC Loss 训练

# 推理（Phase 4 实现后）
python src/inference.py       # 摄像头实时识别
```

## v1 约束

**明确不做**：语言模型校正、beam search、MediaPipe Holistic、近形手势混淆处理。
BiLSTM 双向 ~150ms 延迟可接受，不换单向。YOLO 旁路可关闭。

## 编程规范

- 代码放 `src/`，不混入 TFNet-main/ 等第三方目录
- 路径用 `D:/red star project/...` 绝对路径或项目相对路径
- batch_size=1（序列长度不一致），标签 Gloss 按 `/` 分割，空项过滤
- 每完成 Phase/子任务必须更新 `CHANGELOG.md` 和 `CLAUDE.md`
- 多轮思考无进展时立即停止报告，不反复尝试相同方案
- 不创建 CLAUDE.md 以外的 .md 文件（除非用户明确要求）
- 新增依赖必须记录到 `requirements.txt` 和 `CHANGELOG.md`
- 外部启发先入 `.claude/INSPIRATION.md`，不直接实现

## 已知错误模式（Landmines）

新会话容易踩的坑：

| 陷阱 | 错误认知 | 正确事实 |
|------|---------|---------|
| TFNet 复用 | "复用了 BiLSTM.py/Train.py/DataProcessMoudle.py" | 只有 WER.py 被复用，模型/训练/数据加载均独立实现 |
| MediaPipe API | "用 `mp.solutions.hands`" | 已迁至 `mp.tasks.vision.HandLandmarker`，IMAGE 模式 |
| 词表大小 | "~500-1000 词" | 3515 tokens（`vocab.json`） |
| 模型参数 | "~5M" | 13.1M |
| 视频路径 | "CE-CSL/video/" | 实际在 `CE-CSL/CE-CSL/video/{train,dev,test}/{A-L}/` |
| 坐标归一化 | "预处理已做手腕归一化" | 未实现，Phase 1 存的是原始坐标 |
| ctc_decoders | "C++ 解码库可用" | 存在但未编译，目前用自写贪心解码 |
| 复用决策 | "能复用的就复用" | 删比重写更费劲就不复用 |

## 代码参考索引

写代码时的速查表。设计讨论见 `.claude/INSPIRATION.md`。

| 要写的模块 | 参考项目 | 关键文件:行号 | 参考点 |
|-----------|---------|-------------|--------|
| **model.py** | TFNet-main | `Module.py:25-70` | TemporalConv + update_lgt 长度追踪 |
| | wenet-main | `encoder.py:302-362` | forward_chunk_by_chunk cache 机制 |
| **dataset.py** | TFNet-main | `DataProcessMoudle.py:435-473` | collate_fn stride-aware pad |
| **decode.py** | wenet-main | `search.py:64-101` | CTC PrefixScore 增量维护 |
| | espnet-master | `ctc_prefix_score.py:12-361` | prefix beam search |
| **preprocess_keypoints.py** | landmark-extraction | `extractor.py:134-153` | NaN 填充缺失手部 |
| | landmark-extraction | `analyze.py:49-101` | NaN% 缺失率分析 |
| **Phase 4: preprocess.py** | landmark-extraction | `extractor.py` | MediaPipe 封装模式 |
| **Phase 4: window.py** | wenet-main | `encoder.py:302-362` | strided chunk 滑动窗口 |
| **Phase 4: inference.py** | wenet-main | `speech_client.py:62-147` | StreamingSpeechClient 主循环 |
| **WER 评估** | TFNet-main | `WER.py` | 直接 import（唯一复用） |
| **config.yaml** | espnet-master | `conf/train_asr_streaming_conformer.yaml` | YAML 驱动配置范例 |

使用方式：打开参考文件指定行号，理解核心思路，**用自己的变量名和约定重写**，不复制粘贴。`reference/` 为只读，禁止修改。

## 决策记录格式

每次非显而易见的工程决策，在 CHANGELOG.md 记录**为什么**：

```
# 正确
MediaPipe IMAGE 模式替代 VIDEO 模式：VIDEO 每 ~100s 触发 Google 遥测超时 ~45s（被墙），
实测 IMAGE vs VIDEO 一致性 99.5%，选 IMAGE。

# 错误
MediaPipe 切到 IMAGE 模式。
```

## 冲突裁决

代码 > CHANGELOG > ARCHITECTURE.md > IMPLEMENTATION.md。CLAUDE.md 与代码/CHANGELOG 矛盾时以代码为准，立即修正 CLAUDE.md。
