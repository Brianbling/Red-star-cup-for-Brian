# 红星光项目 (Red Star Project)

## 目标
中国手语（CSL）实时连续识别。摄像头视频流输入 → 手语文本实时输出。

## 系统架构
详见 `docs/permanent/ARCHITECTURE.md`。核心管线：

```
摄像头 → 统一预处理 → MediaPipe Hands (42点关键点)
                      │
                      ├─ 主路: 置信度清洗 → 坐标归一化 → 滑动窗口(3s)
                      │        → 1D Conv(stride=2×2) + BiLSTM(2层,双向) → CTC解码 → 后处理去重 → 文本
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
**现状**：`docs/temporary/实验方案.md`（三阶段路线：SLR 25K 孤立词预训练 → CSL-Daily 20K 数据扩展 → 视觉评估）存在于主仓库但 **git 未跟踪**（仅 `.gitkeep` 入库）。该文件仍是当前最权威的数据扩展路线图，但如需在 worktree 中修改请先 `git add` 或直接在 main 仓库改。

## 目录结构

```
D:/red star project/
├── CLAUDE.md                 # 项目入口（本文件）
├── README.md                 # 项目概览
├── requirements.txt          # Python 依赖
├── vocab.json                # 词表（build_vocab.py 生成，3515 tokens）
├── vocab_top478.json         # 子词表（top-478 高频词 + blank = 479 tokens，快速验证用）
├── docs/                     # 项目文档（除 CLAUDE.md 外所有 md）
│   ├── permanent/            # 永久保留，始终与代码同步
│   │   ├── ARCHITECTURE.md   # 系统架构设计文档（权威参考）
│   │   ├── IMPLEMENTATION.md # 实现计划
│   │   ├── INSPIRATION.md    # 23 条灵感、优先级投票
│   │   └── CHANGELOG.md      # 工作日志，持续追加
│   └── temporary/            # 实验路线/诊断，过期后删除
│       └── 实验方案.md       # 三阶段路线（孤立词预训练 → 数据扩展 → 视觉评估）
├── src/                      # 项目源代码
│   ├── build_vocab.py        # 词表构建
│   ├── preprocess_keypoints.py # Phase 1: 视频 → 关键点 .npy（含 normalize_hand 手腕归一化）
│   ├── model.py              # 1D Conv(stride=2×2) + BiLSTM(2层,双向=512) + Linear + LogSoftmax + visual_fusion
│   ├── dataset.py            # KeypointDataset + collate_fn(time-major pad)
│   ├── train.py              # CTC Loss + blank penalty + 熵正则训练脚本（含 --visual-fusion/--group-size/--augment/--beam-width）
│   ├── decode.py             # CTC 贪心解码 + 后处理（含 ctc_prefix_beam_search）
│   ├── activity_detect.py    # 切掉手部丢失帧（训练时）
│   ├── vocab_utils.py        # clean_word() 标签清洗（vocab 和 dataset 共享）
│   ├── augmentation.py       # 时序增强（实验证明无效）
│   ├── normalize_existing.py # 离线复归一化已有 .npy（8 进程，5987 文件 ~10s）
│   ├── beam_quick_test.py    # beam search 快速验证（10 样本）
│   ├── validate_beam.py      # beam search 全量验证脚本
│   ├── extract_visual_features.py  # MobileNetV3-Small 视觉特征提取（已否决）
│   ├── extract_isolated_words.py   # 孤立词关键点提取（手腕归一化）
│   ├── build_isolated_index.py     # 孤立词→vocab 匹配索引
│   ├── isolated_dataset.py         # 孤立词数据集 + CombinedDataset
│   ├── csldaily_dataset.py         # CSL-Daily 数据集（读 keypoints + labels.json，可混入训练）
│   ├── visual_backbone.py    # 视觉 backbone（MobileNetV3 封装，实验用）
│   ├── video_dataset.py      # 视频数据集（视觉实验用）
│   ├── model_visual.py       # 视觉融合模型（实验用）
│   └── train_visual.py       # 视觉实验训练脚本（实验用）
├── TFNet-main/               # 原 TFNet，仅复用 WER.py（其余均独立实现）
├── CE-CSL/CE-CSL/            # 中国手语连续句子数据集（主路时序模型训练，数据实际在 E:/CE-CSL/CE-CSL）
│   ├── video/{train,dev,test}/  # ~6000 条视频 (.mp4)，train-01418 缺失
│   ├── label/{train,dev,test}.csv
│   └── keypoints/{train,dev,test}/  # 预处理关键点缓存 (.npy)，5987 文件
├── CSL_basic_dataset/         # 中国手语基础词，235 mp4
├── CSL_common_dataset/        # 中国手语常用词，863 mp4
├── CSL-Daily/                 # 中国手语连续句数据集（20,653 关键点 npy，本项目 MediaPipe 提取），实际在 E:/CSL-Daily/CSL-Daily
├── SLR_Dataset/               # CSL-2015，25K 孤立词 + 100 句连续（含 keypoints/ 缓存，仅 000-189 类已提取）
├── isolated_words/           # 孤立词关键点提取缓存（basic/common，手腕归一化），1057 npy + index.json（254 token / 271 样本），gitignore 不跟踪
├── YOLOv8/                   # YOLOv8，含 l1290_data.yaml + train_l1290.py；L1290 权重在 runs/l1290/weights/best.pt
├── ASL Alphabet/             # 美式手语字母数据集（尚未下载，需使用时再获取）
├── L1290/                    # C382 手势手语数据（35 类，2148 训练图，YOLO 检测格式）
├── ctc_decoders-master/      # CTC beam search + 贪心解码 C++ 库（未编译）
├── models/                   # 模型文件
│   └── hand_landmarker.task  # MediaPipe Hand Landmarker (~7.6MB)
├── checkpoints/              # 模型权重保存目录
│   ├── best.pt               # 当前最佳 WER 权重（66.58%）
│   ├── last.pt               # 最近 epoch 权重
│   ├── csldaily_iso/         # 孤立词混合训练（best 64.24%，2026-08-02）
│   └── exp_*/                # 各实验独立 checkpoint（kp_only/exp1_bs4/exp2_augment/exp3_beam 等）
├── backend/                  # Web 后端（FastAPI + WS，S1 起）
├── inference/                # 推理封装（static_yolo.py 等，S1 起）
├── frontend/                 # Web 前端（index.html + app.js，S1 起）
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
- `src/model.py` — 1D Conv（两个 stride=2，总降采样 /4，84→256）+ BiLSTM(2层,双向,hidden=512) + Linear（独立实现，未复用 TFNet BiLSTM.py）
- `src/train.py` — CTC Loss 训练脚本（独立实现，未复用 TFNet Train.py）
- `src/dataset.py` — KeypointDataset + collate_fn（读 .npy，非原始视频帧，未复用 TFNet DataProcessMoudle.py）
- TFNet `WER.py` — 词错误率评估（唯一复用的 TFNet 模块）
- ctc_decoders（C++ CTC 解码，含贪心和 beam，有 SWIG Python 绑定，未使用）

### 旁路（静态手势）— 数据齐全 + 权重已训
- CSL_basic_dataset（235 个中国手语单词视频，文件名即标签）
- CSL_common_dataset（863 个中国手语常用词视频，文件名即标签）
- 注意：两个数据集均无 CSV 标签文件，无 train/test 划分，需自行 8:2 划分
- 注意：视频为词级（非帧级标注），YOLO 分类训练时从视频抽帧，同视频所有帧共享文件名词类
- ASL Alphabet 数据集（字母手势，可作预训练补充；尚未下载，目录不存在）
- **L1290 静态手势分类已训练完成**：yolov8s + COCO 预训练，100 epochs，最优 mAP50=0.985（epoch 18）、mAP50-95=0.805（epoch 55），权重 `YOLOv8/runs/l1290/weights/best.pt`。**坑**：`YOLO("yolov8s.yaml").train(pretrained=True)` 不会真正加载预训练权重，必须传 `.pt` 文件（详见 CHANGELOG 2026-08-01）
- 旁路尚未集成到实时推理管线（Phase 5 集成部分待做，YOLO 分类模型本身已就绪）

## v1 约束

**明确不做**：语言模型校正、beam search、MediaPipe Holistic、近形手势混淆处理。
BiLSTM 双向 ~150-300ms 延迟可接受，不换单向。
YOLO 旁路可关闭，低算力设备纯时序识别。
beam search 已实测与贪心等价（WER 0pp 变化，P(blank)≈0.77 概率分布太尖锐），不再尝试。

## 训练流程（离线）

```
CE-CSL 视频 → MediaPipe Hands 逐帧提取关键点 → .npy (每视频一个文件)
→ Dataset/DataLoader → 1D Conv(stride=2×2) + BiLSTM(2层,双向) + CTC Loss → best.pt
```

## 推理流程（实时）

按 `docs/permanent/ARCHITECTURE.md` 管线实现。核心模块：
1. `preprocess.py` — 预处理 + MediaPipe Hands + 置信度清洗 + 坐标归一化
2. `window.py` — 滑动窗口（3s/90帧 deque）
3. `arbitration.py` — 融合仲裁状态机
4. `model.py` — 1D Conv(stride=2×2) + BiLSTM(2层,双向) + Linear
5. `decode.py` — CTC 贪心解码 + 后处理
6. `inference.py` — 摄像头主循环，串起以上模块

## 编程规范

- 新代码放项目根目录，不混入 TFNet-main/
- 路径用 `D:/red star project/...` 绝对路径或项目相对路径
- 推理和训练代码分开
- batch_size 默认 2（train.py），分组 padding (bs=4) 实验证明退步，不要尝试大 batch
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
| 模型架构 model.py | **自己写** | ~70 行 | 1D Conv(stride=2×2) + BiLSTM(2层,双向) + Linear + visual_fusion 模式 |
| Dataset/DataLoader | **自己写** | ~90 行 | 读 .npy + 标签解析 + collate_fn |
| 训练脚本 | **自己写** | ~250 行 | 独立实现，未复用 TFNet Train.py |
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
| 2 | 主路模型训练（1D Conv(stride=2×2) + BiLSTM + CTC） | 12-24h | GPU，挂机 |
| 3 | CTC 解码（贪心 + 后处理去重） | 1-2h | CPU |
| 4 | 实时推理管线（6 个模块 + 联调） | 4-6h | CPU |
| 5 | 旁路 YOLO 分类（数据准备 + 训练 + 集成） | 6-10h | GPU |
| 6 | 系统联调测试（功能 + 延迟 + 场景） | 3-4h | CPU + 摄像头 |

**执行顺序**：Phase 0 → Phase 1（挂机）→ Phase 2（挂机，同时做 Phase 3）→ Phase 4 → Phase 5 → Phase 6

## 当前进度

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 环境搭建 | **已完成** (2026-07-28) |
| 1 | 关键点预处理 | **已完成** (2026-07-28，5987 .npy，含归一化) |
| 2 | 模型训练 | **实验全部完成** (2026-08-01)，WER=66.58%，~5K 数据是唯一瓶颈 |
| 3 | CTC 解码 | 已完成（src/decode.py，贪心解码 + blank penalty + 熵正则；beam search 验证与贪心等价） |
| 4 | 实时推理管线 | 待开始 |
| 5 | 旁路 YOLO | **分类模型已训好**（L1290 mAP50=0.985），集成到推理管线待做 |
| 6 | 联调测试 | 待开始 |

**Phase 2 实验结论（2026-07-28 → 2026-08-01）**：
- 当前最优：全量 kp-only + activity detection，top-478 子词表，best WER **66.58%**（Epoch 52，D=41.8% S=21.3% I=3.5%）
- **WER 瓶颈 = 数据量（~5K 样本）**，不是模型架构（BiLSTM 13.5M）、特征维度（84d）、解码策略（贪心）、归一化或标签完整性
- 已排除的假设（详见 CHANGELOG + Landmines）：blank 坍塌、视觉特征融合、activity detection、clean_word 标签 bug、分组 padding、时序增强、beam search、归一化缺失
- 已交付的数据扩展：孤立词 1098 .npy + index.json（94 匹配视频，CombinedDataset 已实现可混入 CE-CSL train）；L1290 YOLO 旁路权重（mAP50=0.985）
- 下一步（`docs/temporary/实验方案.md`，主仓库未跟踪）：SLR 25K 孤立词预训练 + CSL-Daily 20K 数据扩展

**2026-08-02 数据扩展实验更新（进行中）**：
- **CSL-Daily 混合训练**：`--csldaily-base E:/CSL-Daily/CSL-Daily` 混入 18,400 连续句，数据量 4.7 倍，best WER **66.76%**（Epoch 69）——与 baseline 66.58% 几乎持平
- **三路诊断修正了"数据量是唯一瓶颈"结论**：真实瓶颈是三层叠加——① 词汇天花板（CSL-Daily 663 个 OOV 词占 9.5% token 被静默丢弃成负信号；dev 22.4% token 是 train 低频词）、② 缺手率域偏移（CE 30.1% vs CSL-Daily 2.55%，CSL-Daily 干净帧稀释 CE 的"无手→blank"能力）、③ 孤立词索引缺口（isolated_words 1057 npy 但 index.json 只收 87 token，167 个在词表内的词从未被训练使用）
- **孤立词索引已修复**（index.json 87→254 token / 271 样本）+ train.py 加 `--isolated-index` 参数
- **新训（isolated 混合）跌破 baseline**：`checkpoints/csldaily_iso/`，best WER **64.24%**（Epoch 77/81，S=20.1% D=42.4% I=1.7%）。验证"词汇覆盖是瓶颈"（孤立词补 S 分类能力，D 回 baseline 持平）。**注意口径**：64.24% 是完整 3515 词表（dev 2455 token），66.58% baseline 是 top-478 子词表（dev 1954 token，丢 20.4% OOV token）——两者**不可直接相减**，真实改善下界 ≥9.16pp（可比的正确 baseline 是 66.76% csldaily_vae → 差 2.52pp）。详见 2026-08-02 代码审查结论

**Web 化前后端识别系统（2026-08-02 启动）**：架构详见 `ARCHITECTURE.md` v2 章节。S1 先接旁路 YOLO（`l1290 best.pt`，35 类）验证训练成果 → S2 接主路 BiLSTM+CTC（顺带完成 Phase 4）→ S3 仲裁融合 + config.yaml。技术栈：FastAPI + WebSocket + 浏览器 getUserMedia。文件：`backend/` `inference/` `frontend/`。新增依赖 fastapi/uvicorn/websockets。**注意**：YOLO 权重实际在 worktree `.../exp+zero-cost-optimization/YOLOv8/runs/l1290/weights/best.pt`（主仓库无此路径）

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
| 模型参数 | "~5M" | 13.5M（vocab=3515，BiLSTM 2 层 512d）；子词表 vocab=479 时 10.4M（实测 13,515,451 / 10,403,551） |
| 视频路径 | "CE-CSL/video/" | 实际在 `CE-CSL/CE-CSL/video/{train,dev,test}/{A-L}/`（数据已迁移至 E:/CE-CSL/CE-CSL） |
| 复用决策逻辑 | "能复用的就复用" | **删比重写更费劲就不复用。** 不为了"遵守计划"而制造垃圾代码 |
| 归一化已实现 | "main 分支 `src/preprocess_keypoints.py` 仍是原始坐标版本" | E: 训练数据早已归一化（wrist 原点 + 腕→中指根缩放，1,247,783 帧 0 例外），归一化不是瓶颈。`src/preprocess_keypoints.py` 已合入 `normalize_hand()`（2026-08-01）。66.58% baseline 本就在已归一化数据上训练 |
| ctc_decoders | "C++ CTC 解码库可用" | 存在但未编译/未使用，目前用自写贪心解码 |
| CTC blank 坍塌 | "loss 下降 = 模型在学习" | `zero_infinity=True` 丢弃 inf batch，loss 表面下降但模型输出全 blank。长序列下 T/L 平均 ~34:1（实测 dev 数据，极端样本 50:1+），blank 是 CTC 最便宜路径 |
| SR-CTC 能救 blank | "CR-CTC 论文的 SR-CTC 能压制 blank" | KL 力差 ~500x（0.01 vs 6.0），拦不住。SR-CTC 是辅助正则项，不是 blank 坍塌的银弹 |
| FC bias 反 blank | "给 blank 大负 bias 就能压制" | 压过头（-4.0）模型死锁，WER=100% 永远不动。CTC 需要 blank 做分隔符，不能完全杀死 |
| Activity Detection | "去掉手部丢失帧，WER 就大幅下降" | collapse% 从 31.2% 降到 9.4%（-70%），但 WER 不变（66.5%→66.85%）。坍缩样本不是 WER 瓶颈，剩余错误是 token 预测错误 |
| Visual Features | "MobileNetV3-Small 576d 特征能提升手语识别" | ImageNet 预训练特征编码物体类别（猫、车），与手语无关。Raw concat WER 84.79%（collapse=0%），比 kp-only 66.58% 差 18pp。Projected fusion 更差（91.61%），因噪声获得了不对等建模容量。collapse 消除但 token 预测错误取代了它 |
| clean_word 一致性 | "build_vocab 和 dataset 用同一套清洗逻辑，标签一定对得上" | `build_vocab.py` 调 `clean_word()` 去括号再存词表，`dataset.py` 只 `w.strip()` 就查 word2idx → KeyError → 静默跳过。含括号 token 在训练中丢失。已修复（`vocab_utils.clean_word`），但重训后 WER 不变（66.85%→66.58%），标签完整性不是当前瓶颈 |
| WER.py D/I 交换 | "WerList 返回的 del_rate/ins_rate 是准确的" | 编辑距离初始化和 backtrace 中 D/I 标签交叉绑定。总 WER 不受影响（代价均为 1），但 `del_rate` 和 `ins_rate` **互换**。已修复：S/D/I 均用正确语义 |
| Grouped Padding | "分组 padding(bs=4) 减少填充量，能提升 WER" | WER 退步 2.3pp（66.58%→68.83%）。batch 变小导致梯度噪声大，序列长度差异本身是有效正则化。不要尝试改 bs |
| Temporal Augmentation | "坐标增强 + jitter 能提升泛化" | WER 退步 9.1pp（66.58%→75.69%）。84d 低维特征空间下，坐标级增强噪声大于信号。~5K 数据不够，增强放大方差 |
| Beam Search (CTC Prefix) | "beam search 比贪心好 5-8pp" | WER 0pp 变化。模型 P(blank)≈0.77，概率分布过于尖锐，无备选路径可探索。beam search 只在模型足够"犹豫"时有价值 |
| blank_bias=5.32 | "初始化给 blank 正 bias 不合理" | 不给正 bias（bias=0）时 P(blank) 从 0→1.0 仅需 1 epoch。bias=5.32 是正确初始化，不是 hack。CTC 需要 blank 做分隔符，bias 控制初始 P(blank) ≈ σ(5.32)≈0.995 |
| 数据瓶颈 | "改进训练/解码策略就能突破 WER" | 三项零成本优化均退步或不改善。~5K 数据量是 WER 天花板（66.58%），突破需要更多数据或预训练，非工程优化 |
| 数据量翻倍 = WER 提升 | "4.7 倍数据（CSL-Daily 混合）应显著降 WER" | 实测 66.58%→66.76% 几乎持平。**瓶颈是词汇覆盖 + 缺手率域偏移，不是样本数**：CSL-Daily 在词表内的 1337 词全被 CE-CSL 覆盖（0 个新词），663 个 OOV 词占 9.5% token 被静默丢成负信号；dev 22.4% token 是 train 低频词（CSL-Daily 帮不上）|
| CSL-Daily OOV 静默丢弃 | "`_gloss_to_ids` 查不到就跳过，无害" | 标签被丢但手势帧还在，模型被教成"这段手势→blank"（负信号）。663 个 OOV 词占 9.5% token |
| 缺手率域偏移 | "两个数据集同管线提取，分布一致" | 唯一显著偏移是缺手率：CE 30.1% vs CSL-Daily 2.55%（12x）。CSL-Daily 干净帧主导梯度，稀释 CE 学"无手→blank"，混合训练后 D 从 41.8%→46.4% 退步 |
| isolated_words 索引 | "isolated_words 只覆盖 87 词，数据不全" | **数据全在**（1057 npy：basic 235 + common 863 全量），但 index.json 只收 87 token，**167 个在词表内的词从未被训练使用**（含 dev 8 个 ≤3 次最难词）。2026-08-02 已重建 index.json（254 token / 271 样本）。新会话别再说"孤立词只有 87 词" |
| L1290 YOLO 是时序模型 | "L1290 是手势时序分类，与主路同思路" | L1290 是 **YOLO 静态手势检测/分类**（35 类，2148 图），训练产物是旁路静态词权重 `YOLOv8/runs/l1290/weights/best.pt`（mAP50=0.985），与主路 BiLSTM+CTC 完全独立 |
| YOLO 预训练加载 | "`YOLO('yolov8s.yaml').train(pretrained=True)` 会加载 COCO 权重" | **不会**。yaml 构建的模型无 ckpt，bool pretrained 不触发 load_checkpoint（log 无 "Transferred" 行）。必须 `YOLO("yolov8s.pt")`。GitHub 下载 SSL 失败需 `ssl.CERT_NONE` + urllib 手动下载 |
| 停服务/杀进程 | "重启服务用 `taskkill //F //IM python.exe` 全杀很省事" | **禁止按进程名全杀**（`//IM`/`taskkill python.exe`/`pkill`）。会连带杀掉同名的模型训练进程（2026-08-02 事故：S1 验证时误杀 CSL-Daily 训练 PID 50848，Epoch 87/100 中断）。停服务必须用**精确 PID**（`Stop-Process -Id <pid>` / `taskkill //F //PID <pid>`），先 `Get-CimInstance Win32_Process` 查准 PID 再杀 |
| 平板/局域网摄像头 | "平板连 `http://192.168.x.x:8000` 就能用摄像头" | getUserMedia 要求 **secure context**（HTTPS 或 localhost），局域网 http 会被浏览器拦截摄像头。必须 HTTPS（自签证书，平板接受警告）。S1 用 `https://192.168.1.8:8000` |
| build_isolated_index 词表硬编码 | "重跑 build_isolated_index.py 无害，只是重建索引" | 2026-08-02 前版本硬编码 `vocab_top478.json`，重跑会把 ISO 从 254 token/271 样本缩到 87/94（丢 dev 7 个 ≤3 次最难词），与 train.py `--vocab vocab.json` 脱节。**已修复**：默认改用 `vocab.json` + `--vocab` 参数。重跑后校验输出应为 254/271 |
| 跨词表 WER 直接相减 | "64.24% vs 66.58% = 提升 1.81pp" | **无效比较**。66.58% 是 top-478 子词表（dev reference 1954 token，丢 501 个 OOV token=20.4%，真实全词表 WER 下界 73.4%），64.24% 是完整 3515 词表（dev 2455 token）。真实改善下界 ≥9.16pp；可比 baseline 是 66.76%（同为 3515 词表）→ 差 2.52pp。**跨 checkpoint 比较必须用同一词表 + 完整 dev reference** |
| 评估口径（activity_detect 裁剪 dev） | "报告 WER 64.24% 可直接与 TFNet 42.1% 比" | dev 评估套用了训练侧的 activity_detect，裁剪 32.6% 帧（411/515 样本）→ 同权重全 dev WER 实为 **69.49%**（+5.2pp）。TFNet 42.1% 是未裁剪全视频口径，真实差距 ~27.4pp 而非 22pp。方向是让我们数字更好看 |
| 长度对齐 `L//4` | "`input_lengths=(L//4)` 与 conv 实际输出一致" | conv 输出是 `ceil(L/4)`，`//4` 在 L%4≠0（dev 72.8% 样本）时丢 1 帧尾帧。WER 影响 ~0.2pp 且实测 ceil 反而略差，train/eval 一致使用故非 train/infer 错位。属低收益清理项，可留文档备注 |
| normalize_hand scale 兜底 | "scale<1e-6 才兜底 0.1 够稳" | 1e-6~0.08 的小手尺度会放大坐标 12~650 倍（实测 train-00898 coord_max=651.66）。但尖峰帧仅 0.33% 有效手帧且 dev/test 对称，LayerNorm+LSTM 已吸收，WER 影响低-中。建议加 scale 真实下限 `max(scale, 0.02)` |
| 数据量 4.7x | "CSL-Daily 18,400 文件 = 4.7 倍数据" | 18,400 文件实为 **6,598 distinct 句子**（平均 2.8 段/句，P 是 signer ID 非重复镜头），真实 distinct 内容 ~2.4x。引用数据量时须注明 distinct 句数 |

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
