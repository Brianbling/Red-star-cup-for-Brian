# 工作日志

## 2026-07-28

### Phase 0 — 环境搭建 [完成]
- 创建目录 `checkpoints/`、`CE-CSL/CE-CSL/keypoints/{train,dev,test}/`
- 验证依赖：torch 2.7.1+cu128 (GPU)、mediapipe 0.10.35、opencv 5.0.0、numpy 2.2.6、ultralytics 8.4.105
- 生成 `requirements.txt`
- 更新 CLAUDE.md

### Phase 1 — 关键点预处理 [完成]
- **API 变更**：MediaPipe 0.10.35 用 `mp.tasks.vision.HandLandmarker`（旧版 `mp.solutions.hands` 已移除）
  - `visibility` → `presence`，`detect_for_video(image, timestamp_ms)` 替代 `detect()`
  - 需单独下载 `.task` 模型文件（`models/hand_landmarker.task`，~7.6MB）
  - **clearcut 遥测问题**：VIDEO 模式每 ~100s 触发 Google 遥测超时（~45s/次，被墙），改为 IMAGE 模式（`detect()` 无时间戳）+ 全局复用单个 HandLandmarker，消除阻塞
  - IMAGE vs VIDEO 模式对比：检测一致性 99.5%，坐标差异 0.012，无遥测阻塞
  - 实测 ~63ms/帧 CPU，单进程全量 5988 视频约 19h
- 创建 `src/preprocess_keypoints.py`，支持断点续跑
- 输出格式：(T, 84) float32 → `CE-CSL/CE-CSL/keypoints/{split}/{video_id}.npy`
- **多进程重写**：`multiprocessing.Pool` + 每 worker 独立 HandLandmarker + `imap_unordered`。32 进程 16GB OOM → 8 进程安全
- **全量完成**：train 4972/4973 (缺 train-01418)，dev 515/515，test 500/500。合计 5987 .npy，~0.4 GB
- 清理冗余：删除 `mediapipe-master/`、`hagrid-master/`

### Phase 2/3 — 模型 + 训练 + 解码 [代码完成，待训练]
- **`src/build_vocab.py`**：从 CE-CSL CSV 构建词表 → `vocab.json`（3515 tokens）
- **`src/model.py`**：1D Conv(84→256) + BiLSTM(256→512, 2层) + Linear(→vocab_size) + LogSoftmax，13.1M 参数
- **`src/dataset.py`**：KeypointDataset 读取 .npy + CSV 标签，collate_fn time-major pad
- **`src/train.py`**：CTC Loss + Adam + ReduceLROnPlateau + early stopping(patience=10) + WER 评估（复用 TFNet WER.py）
- **`src/decode.py`**：CTC 贪心解码（argmax → unique_consecutive → 去 blank）
- 所有模块导入、shape、解码流程验证通过

### 文档维护
- **目录结构**：所有项目代码迁入 `src/`，与第三方库和数据分离
- **ARCHITECTURE.md 修复**：坐标归一化分母"双肩宽度"→"手腕间距"（MediaPipe Hands 无肩膀点）
- git init + .gitignore + 首次 commit

### 2026-07-28 (晚)
- **Phase 1 进度**：train 1796/4972 (36.1%)，正常运行中
- 清理无关进程：Anaconda Navigator (pythonw.exe x2)
- **文档校正**：ARCHITECTURE.md 训练流程描述（独立实现非复用 TFNet）、IMPLEMENTATION.md 词表 3515（非~500-1000）、模型 13.1M（非~5M）

## 2026-07-29

- **训练计划明确化**：硬约束——先跑通 CE-CSL 6000 句 baseline WER，再碰 CSL-Daily。禁止并行开工
- **新增 reference/**：wenet、espnet、landmark-extraction、slt、slt_how2sign、k2 代码参考项目
- **INSPIRATION.md 大幅扩展**：从 8 条增至 23 条，覆盖 7 个来源。5 位 agent 完成交叉头脑风暴，生成 v1 优先级投票、共识/分歧报告、解码能力升级路线
- **CLAUDE.md 治理增强**：新增代码参考索引（模块→参考文件:行号映射）、Landmines 表、CSL-Daily 三步训练计划
- **文档修正**：CLAUDE.md 代码参考索引标注已完成模块（✓）、ARCHITECTURE.md 修正过时的 TFNet 复用清单（仅 WER.py）、时序模型描述（84维非42维、未复用 BiLSTM.py）
- **新增数据集**：SLR_Dataset（CSL-2015，25K 孤立词 + 100 句连续）、CSL-Daily（E:/，~20K 句，下载中）
- **Phase 1 完成**：train 4972/4973（缺 train-01418），dev 515/515，test 500/500，合计 5987 .npy，~0.4 GB
- **多进程性能**：单进程 ~19h → 8 进程 ~3h。32 进程 16GB OOM → 降至 8 进程后稳定。预处理总帧数约 45 万帧
- **CLAUDE.md 精简整合**：314 行 → ~180 行，删除与 IMPLEMENTATION.md 重复的材料清单/工作量表/时间估算
- **Phase 2 首次训练完成**：Epoch 1-15，early stopping 触发。Best WER 93.15%（Epoch 5），train loss 7.24→5.99，dev loss ~6.3-6.5 不动
  - **结论：baseline 未通过**。缺少坐标归一化（存的是原始像素坐标）+ batch_size=1 + 纯 CTC 贪心解码，模型未学到有效模式
  - **下一步**：坐标归一化（手腕参考点）+ NaN 替代零值（INSPIRATION #15），预处理重跑后再训

### 2026-07-29 — CTC Blank 坍塌诊断与修复（进行中）

#### 背景
P0 修复 + 手部归一化 + top-478 子词表后，训练 WER 始终 ~93-95%（等同随机）。
诊断确认根因：**CTC blank collapse**。

#### 根因：CTC Blank 坍塌
- 99.7% 帧 argmax=blank（训练后模型），初始化时仅 0.2%
- 仅 2 个 unique class/sample（blank + 1 个随机 token）
- 16.1% 样本 zero_pred（空输出）
- Loss 从 9.37→5.35 下降，但 WER 无改善——因为 `zero_infinity=True` 过滤了 inf batch
- **T/L = 68:1**（188 帧 → 3.5 token），blank 是 CTC 的压倒性最便宜路径

#### 已尝试策略（按时间序）

| # | 策略 | 关键参数 | 结果 | 结论 |
|---|------|---------|------|------|
| 1 | 暴露真 loss | `zero_infinity=False` | 发现大量 inf loss | `zero_infinity=True` 是毒药，掩盖了 blank 坍塌 |
| 2 | SR-CTC（CR-CTC 论文 App A.1）| 1D depthwise conv 平滑 + KL divergence，β=0.2 | sr_loss 初版为负值（-1.9） | KL 公式 bug：用了 `probs` 替代 `log_probs` |
| 3 | SR-CTC 修正版 | 修正 KL 公式：`DKL(smooth ‖ original)` | sr_loss=0.0124 vs ctc=6.27 | **KL 力差 ~500x**，完全挡不住 blank 坍塌 |
| 4 | Conv stride=2 降采样 | T 188→94，T/L 68→27 | 仍 blank 坍塌 | 降采样有帮助但单独不够 |
| 5 | Conv stride=4 降采样 | T 188→47，T/L 68→13 | 仍 blank 坍塌 | 结构上方向对，需配合其他手段 |
| 6 | FC bias 极端反 blank | blank bias=-4.0, others=-2.0 | **模型死锁**：11 epoch WER=100%，zero_pred=263/509 不变 | blank 被压太死，CTC 丢失分隔符能力，无对齐路径 |
| 7 | 撤回 bias + stride=2 + SR-CTC | 回到方案 3 | WER=97%，zero_pred=402/509 | 确认 SR-CTC 力不够 |
| 8 | stride=4 + blank penalty 20x | penalty=(blank_prob-0.85).clamp(min=1e-8)·20 | **被中断，未出结果** | — |

#### 关键教训

1. **SR-CTC 的 KL 力不够**：smooth 后分布变化极小（KL≈0.01），不敌 CTC loss（~6.0）。论文中 SR-CTC 是辅助正则项，不是 blank 坍塌的银弹
2. **bias init 是双刃剑**：压过头（-4.0）模型无法学到对齐，CTC 需要 blank 做分隔符
3. **`zero_infinity=True` 是毒药**：inf batch 被丢弃后 loss 表面下降，实际没学到东西
4. **Conformer/Transformer 天然抗 blank 坍塌**（Wenet/ESPnet 经验）：attention 提供更均匀的对齐分布，BiLSTM 在长 T/L 比下尤易塌陷

#### 尚待尝试

- stride=4 + 适度 blank bias（~-0.1，非 -4.0）
- `zero_infinity=True` + stride=4（让 inf batch 跳过而非硬撑梯度爆炸）
- 1D Conv+ResNet 替代当前模型（CSL-Daily 论文做法）
- 先训最长句子（token 比空白多则 blank/T 比更低），而非全量数据
- Conformer/Transformer encoder 替代 BiLSTM

## 2026-07-30

### 根因定位：坐标归一化缺失（8 轮无效实验的根因）

- **8 轮 CTC blank 坍塌实验全部无效**：bias init、SR-CTC、stride 降采样、BiLSTM/ConvResNet/Conformer 均无法阻止坍塌
- **三重诊断定位根因**：
  - 诊断 A：非 blank 帧全集中在序列第一帧（非 CTC 对齐行为，全样本仅 1 个非 blank spike）
  - 诊断 B：全 dev 集 509 样本只输出 7 个 token，91% 集中在"的"+"了"两个高频虚词
  - 诊断 C：自检发现 `preprocess_keypoints.py` 从未实现坐标归一化——所有 5987 .npy 存储原始像素坐标
  - 根本原因：同一个手语动作在不同拍摄距离下产生完全不同的 84 维向量，模型学不到几何不变性，blank 是 CTC 在像素空间的最优解
- **修复**：`preprocess_keypoints.py` 新增手腕参考点归一化（腕→中指MCP距离为分母），每只手独立计算，输出无量纲坐标
  - 对 MCP9（中指根）增加检测失败保护——手腕或 MCP9 为零时整手坐标置零，避免用错误分母归一化
  - Pool initializer 复用 HandLandmarker（每 worker 仅初始化一次），6 进程 ~2h 完成全量
- **效果（归一化前后对比）**：
  - 坐标范围：0-1920 px → [-3, +3] 无量纲
  - 非 blank token 多样性：7 → 55（质的飞跃）
  - zero_pred：41% → 22%（Epoch 24）
  - WER：93-95% → 66.46%（Epoch 24，Conformer，top-478 子词表）
- **P(blank) 拆解分析**：全局 P(blank)=95.7%，但区分帧类型后——P(blank|lost)=96.1%（手部丢失帧，blank 是正确答案），P(blank|valid)=95.6%（有效帧上 T/L=53，CTC 对齐中 blank 天然占比高）。手部丢失帧占 34.2%
- **全量重跑预处理**：删除旧 .npy，6 进程重跑 5987 个视频，约 2h
- **教训**：模型坍塌，先查数据再查架构。数据质量验证（归一化/标签/分布）必须在架构实验之前完成

### 文档更新 (2026-07-30)

- **ARCHITECTURE.md**：归一化描述从"手腕间距"修正为"腕→中指根距离"，补充每只手独立计算的细节和输出范围
- **IMPLEMENTATION.md**：Phase 1 新增坐标归一化实现细节（修复前后对比代码）
- **CLAUDE.md**：Landmines 表新增"不检查数据质量就跳进架构实验"陷阱条目
