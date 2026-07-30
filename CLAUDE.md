# 红星光项目 (Red Star Project)

## 目标
中国手语（CSL）实时连续识别。摄像头视频流输入 → 手语文本实时输出。

## 系统架构
详见 `docs/permanent/ARCHITECTURE.md`。核心管线：

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

## 项目文件索引

根目录只保留 `CLAUDE.md`。其他 Markdown 按生命周期放入 `docs/`。

| 文件 | 类型 | 用途 | 何时读 |
|------|------|------|--------|
| `CLAUDE.md` | 入口 | 治理规则、目录结构、约束 | **每次会话必读** |
| `docs/permanent/ARCHITECTURE.md` | 永久 | 系统架构设计 | 修改模型/管线时 |
| `docs/permanent/IMPLEMENTATION.md` | 永久 | 实现计划 | 设计决策时 |
| `docs/permanent/INSPIRATION.md` | 永久 | 23 条灵感、优先级投票 | 设计决策时 |
| `docs/permanent/CHANGELOG.md` | 永久 | 工作日志，持续追加 | 了解历史决策时 |
| `docs/temporary/` | 临时 | 当前活跃的实验路线/诊断 | 按需读取 |

**规则**：新增临时文档 → 索引加一行。临时文档过期 → 删除文件 + 从索引移除。

## 目录结构

```
D:/red star project/
├── CLAUDE.md                 # 项目入口（本文件）
├── README.md                 # 项目概览
├── requirements.txt          # Python 依赖
├── vocab.json                # 词表（build_vocab.py 生成，3515 tokens）
├── vocab_top478.json         # 子词表（top-478 高频词，快速验证用）
├── docs/                     # 项目文档（除 CLAUDE.md 外所有 md）
│   ├── permanent/            # 永久保留，始终与代码同步
│   │   ├── ARCHITECTURE.md   # 系统架构设计文档（权威参考）
│   │   ├── IMPLEMENTATION.md # 实现计划
│   │   ├── INSPIRATION.md    # 23 条灵感、优先级投票
│   │   └── CHANGELOG.md      # 工作日志，持续追加
│   └── temporary/            # 实验路线/诊断，过期后删除
│       ├── 2026-07-29-ctc-blank-repair-plan.md  # CTC blank 坍塌修复实验路线
│       └── 2026-07-29-ctc-blank-experiment-log.md  # 实验记录
├── src/                      # 项目源代码
│   ├── build_vocab.py        # 词表构建
│   ├── preprocess_keypoints.py # Phase 1: 视频 → 关键点 .npy
│   ├── model.py              # 1D Conv(stride=2×2) + Conformer(4层) + Linear + LogSoftmax
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

按 `docs/permanent/ARCHITECTURE.md` 管线实现。核心模块：
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
- **阅读外部项目/论文/代码库发现有价值的架构模式或工程技巧时，记录到 `.claude/INSPIRATION.md`**（启发日志），包含来源、核心思路、好处/代价、适用时机、讨论状态
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
| MediaPipe Hands 推理 | 调 API | ~20 行 | `mp.tasks.vision.HandLandmarker`，IMAGE 模式 |
| YOLOv8n 分类训练 | 调 API | 数据准备 ~100 行 | `ultralytics`, `model.train()` |
| CTC Loss | 调 API | 1 行 | `torch.nn.CTCLoss` |
| BiLSTM | **自己写** | ~30 行 | `nn.LSTM`，未复用 TFNet BiLSTM.py |
| 视频→关键点预处理 | **自己写** | ~200 行 | 循环读帧 + 调 MediaPipe + 存 .npy |
| 模型架构 model.py | **自己写** | ~50 行 | 1D Conv + BiLSTM + Linear |
| Dataset/DataLoader | **自己写** | ~90 行 | 读 .npy + 标签解析 + collate_fn |
| 训练脚本 | **自己写** | ~180 行 | 独立实现，未复用 TFNet Train.py |
| CTC 贪心解码 | **自己写** | ~40 行 | argmax + unique_consecutive + 去 blank |
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

## 当前进度

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 环境搭建 | **已完成** (2026-07-28) |
| 1 | 关键点预处理 | **已完成** (2026-07-28) |
| 2 | 模型训练 | 实验进行中，关注 CHANGELOG.md |
| 3 | CTC 解码 | 代码完成（src/decode.py） |
| 4 | 实时推理管线 | 待开始 |
| 5 | 旁路 YOLO | 待开始 |
| 6 | 联调测试 | 待开始 |

## 工作流治理（防幻觉与决策一致性）

### 1. 权威文档层级

每个新会话开始后，**在做任何代码修改前**，必须先读下列文件建立认知基线：

| 文档 | 角色 | 说明 |
|------|------|------|
| `ARCHITECTURE.md` | **架构权威** | 管线、模块边界、参数默认值 |
| `CLAUDE.md` | **入口指南** | 目录结构、环境、约束、进度 |
| `CHANGELOG.md` | **事实记录** | 发生过什么、为什么这样决定 |
| `IMPLEMENTATION.md` | **计划参考** | 预估但非权威，实际以代码为准 |
| `.claude/INSPIRATION.md` | **外部启发** | 待讨论，未采纳，不要直接实现 |

**冲突裁决规则**：代码 > CHANGELOG > ARCHITECTURE.md > IMPLEMENTATION.md。如果 CLAUDE.md 与代码/CHANGELOG 矛盾，CLAUDE.md 是错的，立即修正。

### 2. 出方案前的检查清单

每个实现方案必须检查：

```
□ 是否违反 v1 约束？（语言模型/beam search/Holistic/近形混淆 → 直接拒绝）
□ 引用的文件/函数/路径是否存在？（不靠记忆，用查找工具验证）
□ 是否重走了已知死胡同？（读 CHANGELOG.md 找类似尝试）
□ 是否在"待讨论"列表里？（读 INSPIRATION.md，未采纳的不动）
□ 修改范围是否超过了任务要求？（只改必要的，不同时重构不相关代码）
```

### 3. 已知错误模式（Landmines — 新会话容易踩的坑）

| 陷阱 | 错误认知 | 正确事实 |
|------|---------|---------|
| TFNet 复用 | "TFNet BiLSTM.py/Train.py/DataProcessMoudle.py 被复用" | 只有 WER.py 被复用，模型/训练/数据加载均独立实现 |
| MediaPipe API | "用 `mp.solutions.hands`" | 已迁移到 `mp.tasks.vision.HandLandmarker`，IMAGE 模式 |
| 词表大小 | "~500-1000 词" | 3515 tokens（vocab.json） |
| 模型参数 | "~5M" | 13.1M |
| 视频路径 | "CE-CSL/video/" | 实际在 `CE-CSL/CE-CSL/video/{train,dev,test}/{A-L}/` |
| 复用决策逻辑 | "能复用的就复用" | **删比重写更费劲就不复用。** 不为了"遵守计划"而制造垃圾代码 |
| 归一化已实现 | "preprocess_keypoints.py 做了手腕归一化" | 未实现，Phase 1 存的是原始坐标 |
| ctc_decoders | "C++ CTC 解码库可用" | 存在但未编译/未使用，目前用自写贪心解码 |
| CTC blank 坍塌 | "loss 下降 = 模型在学习" | `zero_infinity=True` 丢弃 inf batch，loss 表面下降但模型输出全 blank。T/L=68:1 是根本原因 |
| SR-CTC 能救 blank | "CR-CTC 论文的 SR-CTC 能压制 blank" | KL 力差 ~500x（0.01 vs 6.0），拦不住。SR-CTC 是辅助正则项，不是 blank 坍塌的银弹 |
| FC bias 反 blank | "给 blank 大负 bias 就能压制" | 压过头（-4.0）模型死锁，WER=100% 永远不动。CTC 需要 blank 做分隔符，不能完全杀死 |
| Activity Detection | "去掉手部丢失帧，WER 就大幅下降" | collapse% 从 31.2% 降到 9.4%（-70%），但 WER 不变（66.5%→66.85%）。坍缩样本不是 WER 瓶颈，剩余错误是 token 预测错误 |
| clean_word 一致性 | "build_vocab 和 dataset 用同一套清洗逻辑，标签一定对得上" | `build_vocab.py` 调 `clean_word()` 去括号再存词表，`dataset.py` 只 `w.strip()` 就查 word2idx → KeyError → 静默跳过。已修复（`vocab_utils.clean_word` 共享函数），但重训后 WER 从 66.85%→66.58%（几乎不变），标签完整性不是当前瓶颈 |
| WER.py D/I 交换 | "WerList 返回的 del_rate/ins_rate 就是各自含义" | 编辑距离初始化和 backtrace 中 D/I 标签交叉绑定。总 WER 不受影响（代价均为 1），已修复 |
| clean_word 一致性 | "build_vocab 和 dataset 用同一套清洗逻辑，标签一定对得上" | `build_vocab.py` 调 `clean_word()` 去括号再存词表，`dataset.py` 只 `w.strip()` 就查 word2idx → KeyError → 静默跳过。含括号 token 在训练中丢失。已修复（`vocab_utils.clean_word`），但重训后 WER 不变（66.85%→66.58%），标签完整性不是当前瓶颈 |
| WER.py D/I 交换 | "WerList 返回的 del_rate/ins_rate 是准确的" | 编辑距离初始化和 backtrace 中 D/I 标签交叉绑定。总 WER 不受影响（代价均为 1），但 `del_rate` 和 `ins_rate` **互换**。已修复：S/D/I 均用正确语义 |

### 4. 编辑约束

- **不要改 `src/` 以外的已有代码**（TFNet-main/、YOLOv8/ 等第三方代码）
- **不要创建 CLAUDE.md 以外的 .md 文件**，除非用户明确要求
- **新增依赖必须记录到 requirements.txt 和 CHANGELOG.md**
- **每个 commit 后检查 `git status` 确认干净**
- **修改前先读文件，修改后不重读验证**（工具会报错即为失败）
- **不写注释，除非 WHY 不显而易见**
- **不设计文档，不改计划文件，多问"要不要汇报"**
- **外部启发先入 INSPIRATION.md，不直接实现**

### 5. 决策记录格式

每次做出非显而易见的工程决策后，在 CHANGELOG.md 记录 **为什么** 而不是 **是什么**：

```
# 正确
MediaPipe IMAGE 模式替代 VIDEO 模式：VIDEO 每 ~100s 触发 Google 遥测超时 ~45s（被墙），
实测 IMAGE vs VIDEO 一致性 99.5%，选 IMAGE。

# 错误
MediaPipe 切到 IMAGE 模式。
```
