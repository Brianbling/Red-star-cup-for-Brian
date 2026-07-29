# 红星光项目 (Red Star Project)

## 目标
中国手语（CSL）实时连续识别。摄像头视频流输入 → 手语文本实时输出。

## 系统架构
详见 `ARCHITECTURE.md`。核心管线：

```
摄像头 → 统一预处理 → MediaPipe Hands (42点关键点)
                      │
                      ├─ 主路: 置信度清洗 → 坐标归一化 → 滑动窗口(3s)
                      │        → 1D Conv + BiLSTM → CTC解码 → 后处理去重 → 文本
                      │
                      └─ 旁路: YOLO 静态手势分类(每5帧) → 三重AND门控 → 静态词
                               ↑ 可关闭，低算力设备退化为纯时序识别
```

**融合仲裁层**：速度门控 + 防抖计时器 + 三态状态机（动态/静默/缓冲）。

## 目录结构

```
D:/red star project/
├── ARCHITECTURE.md           # 系统架构设计文档（权威参考）
├── CLAUDE.md                 # 项目说明（本文件）
├── IMPLEMENTATION.md         # 实现计划
├── CHANGELOG.md              # 工作日志
├── README.md                 # 项目概览
├── requirements.txt          # Python 依赖
├── vocab.json                # 词表（build_vocab.py 生成，3515 tokens）
├── vocab_top478.json         # 子词表（top-478 高频词，快速验证用）
├── src/                      # 项目源代码
│   ├── build_vocab.py        # 词表构建
│   ├── preprocess_keypoints.py # Phase 1: 视频 → 关键点 .npy
│   ├── model.py              # 1D Conv(stride=2×2) + BiLSTM + Linear + LogSoftmax
│   ├── dataset.py            # KeypointDataset + collate_fn(time-major pad)
│   ├── train.py              # CTC Loss + blank penalty + 熵正则训练脚本
│   └── decode.py             # CTC 贪心解码 + 后处理
├── TFNet-main/               # 原 TFNet，仅复用 WER.py（其余均独立实现）
├── CE-CSL/CE-CSL/            # 中国手语连续句子数据集（主路时序模型训练）
│   ├── video/{train,dev,test}/  # ~6000 条视频 (.mp4)，train-01418 缺失
│   ├── label/{train,dev,test}.csv
│   └── keypoints/{train,dev,test}/  # 预处理关键点缓存 (.npy)，5987 文件
├── CSL_basic_dataset/         # 中国手语基础词，235 mp4
├── CSL_common_dataset/        # 中国手语常用词，863 mp4
├── SLR_Dataset/               # CSL-2015，25K 孤立词 + 100 句连续
├── YOLOv8/                   # YOLOv8，已有 MNIST demo，待改造为静态手势分类
├── ASL Alphabet/             # 美式手语字母数据集
├── L1290/                    # C382 手势手语数据
├── ctc_decoders-master/      # CTC beam search + 贪心解码 C++ 库（未编译）
├── models/                   # 模型文件
│   └── hand_landmarker.task  # MediaPipe Hand Landmarker (~7.6MB)
├── checkpoints/              # 模型权重保存目录
│   ├── best.pt               # 当前最佳 WER 权重
│   └── last.pt               # 最近 epoch 权重
├── reference/                # 代码参考项目（只读，不修改）
│   ├── wenet-main/           # 流式 ASR 生产框架
│   ├── espnet-master/        # 流式 ASR 研究框架
│   ├── landmark-extraction-main/  # 手语关键点提取工具链
│   ├── slt-master/           # 手语翻译项目 (SignJoey)
│   ├── slt_how2sign_wicv2023-wicv23/  # How2Sign 翻译基线
│   └── k2-master/            # FSA/FST 语音识别框架
└── arguement/                # 相关论文（PDF + txt）
```

## 技术栈与环境

| 项 | 版本 | 状态 |
|----|------|------|
| Python | — | ✓ |
| PyTorch | 2.7.1+cu128 | ✓ GPU 可用 |
| MediaPipe | 0.10.35 (tasks API) | ✓ `HandLandmarker`，模型文件 `models/hand_landmarker.task` |
| OpenCV | 5.0.0 | ✓ |
| ctc_decoders | master (C++) | ✓ 含贪心/beam + KenLM scorer |
| ultralytics | 8.4.105 | ✓ YOLO 训练/推理 |

## 材料清单

### 主路（实时识别）— 齐全
- CE-CSL 数据集（6000 条视频 + CSV 标签，train-01418 缺视频需处理）
- MediaPipe Hands（pip 已装，IMAGE 模式，~63ms/帧）
- `src/model.py` — 1D Conv + BiLSTM + Linear（独立实现，未复用 TFNet BiLSTM.py）
- `src/train.py` — CTC Loss 训练脚本（独立实现，未复用 TFNet Train.py）
- `src/dataset.py` — KeypointDataset + collate_fn（读 .npy，非原始视频帧，未复用 TFNet DataProcessMoudle.py）
- TFNet `WER.py` — 词错误率评估（唯一复用的 TFNet 模块）
- ctc_decoders（C++ CTC 解码，含贪心和 beam，有 SWIG Python 绑定，未使用）

### 旁路（静态手势）— 数据齐全
- CSL_basic_dataset（235 个中国手语单词视频，文件名即标签）
- CSL_common_dataset（863 个中国手语常用词视频，文件名即标签）
- 注意：两个数据集均无 CSV 标签文件，无 train/test 划分，需自行 8:2 划分
- 注意：视频为词级（非帧级标注），YOLO 分类训练时从视频抽帧，同视频所有帧共享文件名词类
- ASL Alphabet 数据集（字母手势，可作预训练补充）
- YOLOv8（已有 MNIST 训练链路，可直接改造为静态手势分类）

## v1 约束

**明确不做**：语言模型校正、beam search、MediaPipe Holistic、近形手势混淆处理。
BiLSTM 双向 ~150ms 延迟可接受，不换单向。
YOLO 旁路可关闭，低算力设备纯时序识别。

## 训练流程（离线）

```
CE-CSL 视频 → MediaPipe Hands 逐帧提取关键点 → .npy (每视频一个文件)
→ Dataset/DataLoader → 1D Conv + BiLSTM + CTC Loss → best.pt
```

## 推理流程（实时）

按 `ARCHITECTURE.md` 管线实现。核心模块：
1. `preprocess.py` — 预处理 + MediaPipe Hands + 置信度清洗 + 坐标归一化
2. `window.py` — 滑动窗口（3s/90帧 deque）
3. `arbitration.py` — 融合仲裁状态机
4. `model.py` — 1D Conv + BiLSTM + Linear
5. `decode.py` — CTC 贪心解码 + 后处理
6. `inference.py` — 摄像头主循环，串起以上模块

## 编程规范

- 新代码放项目根目录，不混入 TFNet-main/
- 路径用 `D:/red star project/...` 绝对路径或项目相对路径
- 推理和训练代码分开
- batch_size 固定为 1（序列长度不一致）
- 标签 Gloss 按 `/` 分割，空项过滤
- **每完成一个 Phase/子任务，必须更新 `CHANGELOG.md`（工作日志）和 `CLAUDE.md`（如目录结构/环境版本变化）**，便于后续开发者追溯
- **多轮思考无进展时立即停止，报告遇到的问题**，不要反复尝试相同方案
- **以下规范事项应提示用户，每次完成后更新状态**

## 项目规范待办

| 项 | 触发时机 | 状态 |
|----|---------|------|
| git init + .gitignore + 首次 commit | 现在 | **已完成** (2026-07-28) |
| pyproject.toml（替换 requirements.txt） | 现在 | **待处理** |
| README.md（从 CLAUDE.md 精简"怎么跑"） | Phase 4 完成后 | **已完成** (2026-07-28) |
| 测试（预处理格式/模型 shape/CTC 解码） | 对应模块写完 | 待处理 |
| config.yaml（全局配置，替代路径硬编码） | 模块文件到 3+ 时 | 待处理 |
| CI（GitHub Actions 自动测试） | 准备发布/交接时 | 待处理 |
| Docker（环境打包） | 准备发布/交接时 | 待处理 |
| ruff/black + pre-commit hook | 准备发布/交接时 | 待处理 |

## 实现计划

详见 `IMPLEMENTATION.md`。总计约 30-50h。


### 实际工作量（调 API vs 自己写）

| 模块 | 方式 | 代码量 | 说明 |
|------|------|--------|------|
| MediaPipe Hands 推理 | 调 API | ~20 行 | `mp.solutions.hands` |
| YOLOv8n 分类训练 | 调 API | 数据准备 ~100 行 | `ultralytics`, `model.train()` |
| CTC Loss | 调 API | 1 行 | `torch.nn.CTCLoss` |
| BiLSTM | 复用 TFNet | 0 行 | 直接 import |
| 视频→关键点预处理 | **自己写** | ~150 行 | 循环读帧 + 调 MediaPipe + 存 .npy |
| 模型架构 model.py | **自己写** | ~80 行 | 1D Conv + BiLSTM + Linear，标准层 |
| Dataset/DataLoader | **自己写** | ~120 行 | 读 .npy + 标签解析 + collate_fn |
| 训练脚本 | **自己写** | ~100 行 | 复用 Train.py 框架，CTC Loss 替换 |
| CTC 贪心解码 | **自己写** | ~30 行 | argmax + 去重 + 去 blank |
| 滑动窗口 | **自己写** | ~30 行 | deque 封装 |
| 融合仲裁状态机 | **自己写** | ~150 行 | 纯逻辑，分支多 |
| 推理主循环 | **自己写** | ~150 行 | 摄像头 + 串模块 + 显示 |
| YOLO 数据准备 | **自己写** | ~100 行 | 抽帧 + 生成目录结构 |

**约 30% 调 API/复用，70% 自己写。** 大部分是胶水代码和拼接逻辑，无复杂算法从零实现。

### 时间估算

| Phase | 内容 | 时间 | 算力 |
|-------|------|------|------|
| 0 | 环境搭建（目录 + requirements + 验证） | 0.5h | CPU |
| 1 | 关键点预处理（6000 视频 → .npy） | 4-6h | CPU，挂机 |
| 2 | 主路模型训练（1D Conv + BiLSTM + CTC） | 12-24h | GPU，挂机 |
| 3 | CTC 解码（贪心 + 后处理去重） | 1-2h | CPU |
| 4 | 实时推理管线（6 个模块 + 联调） | 4-6h | CPU |
| 5 | 旁路 YOLO 分类（数据准备 + 训练 + 集成） | 6-10h | GPU |
| 6 | 系统联调测试（功能 + 延迟 + 场景） | 3-4h | CPU + 摄像头 |

**执行顺序**：Phase 0 → Phase 1（挂机）→ Phase 2（挂机，同时做 Phase 3）→ Phase 4 → Phase 5 → Phase 6
