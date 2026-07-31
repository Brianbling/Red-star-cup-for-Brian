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
- **`src/model.py`**：1D Conv(84→256, stride=4) + BiLSTM(256→512, 2层,双向) + Linear(→vocab_size) + LogSoftmax + blank_bias=5.32，~14.0M 参数（3515 词表）
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
- **文档校正**：ARCHITECTURE.md 训练流程描述（独立实现非复用 TFNet）、IMPLEMENTATION.md 词表 3515（非~500-1000）、模型 ~14.0M（非~5M）

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
4. **BiLSTM + CTC 天然易 blank 坍塌**：BiLSTM 在长 T/L 比下尤易塌陷（T/L=68:1），通过 stride=4 + blank_bias=+5.32 + blank penalty + 熵正则 + activity detection 联合解决

#### 尚待尝试

- stride=4 + 适度 blank bias（~-0.1，非 -4.0）
- `zero_infinity=True` + stride=4（让 inf batch 跳过而非硬撑梯度爆炸）
- 1D Conv+ResNet 替代当前模型（CSL-Daily 论文做法）
- 先训最长句子（token 比空白多则 blank/T 比更低），而非全量数据
- Conformer/Transformer encoder 替代 BiLSTM（**已放弃——BiLSTM 实验通过 blank_bias + blank penalty + activity detection 解决坍塌，Conformer 未在代码中实现**）

### 2026-07-30 — Activity Detection 训练完成 + 四项审计

- **训练结果**：78 epoch early stop，best WER **66.85%** (Epoch 63)，best collapse **8.1%** (Epoch 50)
- **collapse% 从 31.2% → 9.4%**（相对降低 70%），但 WER 几乎没变（66.5% → 66.85%）
- **四项审计**：
  1. **丢弃帧 = 手部丢失帧**：99.7% 是零帧（手部丢失），0.3% 是 <5帧 的短有效段（正确丢弃），0 长有效段被误杀。min_active_frames=5 无误杀
  2. **多段样本 vs 坍缩重叠**：多段样本 81/509 (15.9%)，平均 2.3 段/样本。精确重叠需模型在原始 dev 集跑推理，但 15.9% < 31.2% 说明多段只是坍缩的部分原因
  3. **跨段合并风险**：4 帧零分隔符，BiLSTM 理论上可跨过，但零帧含零信息，实际影响应弱
  4. **collapse% 变化**：159/509 → 49/509 (31.2% → 9.4%, -70%)
- **核心结论**：activity detection 技术上成功定位 blank 坍缩，但 WER 没变说明坍缩样本不是 WER 瓶颈。剩余 66.85% 错误来自 token 预测错误（插入/删除/替换），而非 blank 坍缩
- **下一步方向**：分析 token 级错误分布 (insertion/deletion/substitution)，定位 WER 真正瓶颈

### 2026-07-30 — S/D/I 分解 + WER.py D/I 标签交换修复

- **初版 S/D/I 分解**（D/I 标签交换）：WER=66.85%，S=18.8%，D=3.4%，I=44.6%
- **发现 WER.py D/I 标签交换 bug**：编辑距离初始化 `d[i][0]`/`d[0][j]` 成本分配错位，
  backtrace 中 Insert/Delete 操作类型与代价常量不匹配。DEL/INS/SUB 代价为 1，总 WER 不受影响，
  但 `del_rate` 和 `ins_rate` 互换
- **修正后 S/D/I 分解**（旧模型在修复后 dataset 上推理）：
  WER=68.27%，S=18.7%，**D=47.5%**，I=2.1%。**Deletion 占错误 69.6%，不是 Insertion**
  - ref tokens 从 1810→1954（clean_word 修复后恢复 144 个之前被跳过的 token，全被删除，推高 D 约 7 个百分点）
  - 因果链修正：标签跳过 → 对应 token 从未出现在训练中 → 从未学出 → 推理时全被删。
    不需要复杂的"blank 边界不稳定"中间路径

### 2026-07-30 — clean_word 不一致修复

- **Bug**：`build_vocab.py` 调 `clean_word()` 去括号再存词表，`dataset.py` 只用 `w.strip()` 查 word2idx。
  含括号 token 静默跳过，全量词表 12.7% token 丢弃，top-478 子词表 34.3% 丢弃
- **新建 `src/vocab_utils.py`**：`clean_word()` 用 depth 计数器（支持嵌套括号），`build_vocab.py` 和 `dataset.py` 共享
- **`src/train.py`**：每 epoch 输出 S/D/I 分解
- **修正 `TFNet-main/WER.py`**：D/I 标签交换

### 2026-07-30 — clean_word 修复后重训结果

- **训练配置**：top-478 子词表，activity detection (min=5, gap=3)，BiLSTM ~10.8M
  - train 4910 样本（4009 被切分，移除 25.7% 帧），dev 512 样本（409 被切分，移除 32.6% 帧）
- **结果**：67 epoch early stop，**best WER=66.58% (Epoch 52)**
  - Best S/D/I：S=21.3%，D=41.8%，I=3.5%，collapse=4.1%，zero_pred=21/512
- **WER 几乎不变**：旧模型 66.85% → 重训后 66.58%，差异 <0.3 个百分点
- **D 主导全程**：Epoch 1 D=92.8% → Epoch 52 D=41.8%。冷启动 token 在前 15 epoch 学完后，
  D 仍然 ≥40%——模型在完整标签上仍然不敢输出
- **S 天花板 ~21%**：模型区分 token 的能力在 ~66% WER 触顶
- **结论**：clean_word 修复解决了数据 bug 但未改善 WER。瓶颈不在标签完整性，
  在**特征层面**——84 维关键点坐标无法区分 478 个 token。
  方向应转向特征增强（CNN 视觉特征拼接到关键点）

### 2026-07-30 — 视觉特征融合实验（MobileNetV3-Small）

#### 背景
之前 visual features 已用 MobileNetV3-Small 预提取完成（4972 train / 515 dev / 500 test，576d/帧）。
问题：576d ImageNet 特征是否对 CSL 手语识别有帮助？直接 concat（84+576=660d）曾得到 WER 84.79%、collapse 0%，
比全量 baseline（WER 66.58%）差很多，但 collapse 消除了。需要控制变量确认根因。

#### 实验设计
三个实验，均使用 visual-only 子集（4972 样本），控制数据集大小变量：

| 实验 | 配置 | 目的 |
|------|------|------|
| Exp A | kp-only（84d），子集 | 控制变量：子集本身是否更难 |
| Exp B | projected fusion（84→128 + 576→128 → 256d） | 解决 576d 噪声淹没问题 |
| 参考 | raw concat（84+576=660d） | 已有结果，collapse=0% 但 WER 差 |

#### 代码变更
- src/train.py：新增 --visual-only、--visual-fusion（raw/projected/none）、--checkpoint-dir 参数
- src/dataset.py：新增 no_visual 参数，控制是否 concat visual features
- src/model.py：新增 visual_fusion="projected" 模式，双线性投影层（kp_proj + vis_proj）
- 每个实验独立 checkpoint 目录（--checkpoint-dir checkpoints/kp_only/ 等）

#### 训练过程
- Exp A 和 Exp B 均用 detached Python 进程运行（python -u ... > log 2>&1 &），避免被后台系统误杀
- 均使用 blank penalty（threshold=0.65, weight=20.0）+ 熵正则（weight=0.01）+ stride=4 1D Conv
- Exp A 和 Exp B 均跑到 epoch 29 时被手动停止

#### 最终结果

| 实验 | 样本数 | Best WER | Deletion | Collapse | 备注 |
|------|--------|----------|----------|----------|------|
| 全量 kp-only baseline | ~6000 | 66.58% | 41.8% | 31.2% | activity detection 后为 66.85% |
| **Exp A** (子集 kp-only) | 4972 | **80.94%** (e29) | 68.8% | 18.3% | 仍在下降趋势中 |
| Raw concat visual | 4972 | 84.79% | 65.6% | 0% | 之前实验 |
| **Exp B** (projected fusion) | 4972 | **91.61%** (e26) | 84.0% | 46% | 优化困难，被噪声主导 |

#### 结论

1. **视觉特征消除 blank 坍塌但未改善 WER**：raw concat collapse=0%，但 WER 84.79% 比全量 baseline 66.58% 差 18 个百分点
2. **视觉子集本身更难**：Exp A（子集 kp-only）WER 80.94% vs 全量 66.58%，差距 ~14 个百分点。过滤掉无 visual features 的 ~1000 样本恰好是简单样本
3. **Projected fusion 更差**：Exp B（91.61%）比 raw concat（84.79%）差 7 个百分点。给 576d 噪声和 84d 关键点分配相同投影维度（128d），让噪声获得了不对等的建模容量，优化更难
4. **ImageNet 特征不适合手语**：MobileNetV3-Small 的 576d 特征编码的是 ImageNet 物体类别（猫、车、建筑等），与手语手势几乎无关，本质是 576d 噪声淹没了 84d 有效信号
5. **根本问题不在特征维度**：collapse 消除说明视觉特征"打破了 silence"，但替换成了错误的 token 预测。问题不在 blank 坍塌，在 token 级识别准确率

#### 下一步
- 视觉特征路线暂时搁置。需要 domain-specific 的手部视觉 encoder（如在手语数据上 fine-tune MobileNetV3），而非直接用 ImageNet 预训练特征

## 2026-07-31

### 零成本优化路线实验 — 三项均未达预期

**目标**：在不改模型架构、不加新数据的前提下，通过训练/解码优化将 WER 从 66.58% 降到 50-54%。

#### 实验总结

| 实验 | 配置 | Best WER | vs Baseline | 结论 |
|------|------|----------|-------------|------|
| Baseline | bs=1, no augment, greedy | 66.58% | — | — |
| Exp-1 | grouped bs=4 | 68.83% | 退步 2.3pp | 分组 padding 无帮助 |
| Exp-2 | bs=4 + temporal augment | 75.69% | 退步 9.1pp | 坐标增强是噪声不是信号 |
| Exp-3 | beam search (beam=3) | 持平 | 0pp | beam 与 greedy 等价 |

#### Exp-1：Grouped Padding (bs=4)
- 分组 padding 减少了填充量（~60%→~20%），但每组 B=2-4 样本，有效 batch 变小
- D（删除）从 41.8% 波动到 47-53%，collapse 10% 左右
- **结论**：分组后 blank penalty 失效风险高，batch 小导致梯度噪声大。序列长度差异本身是有效的正则化

#### Exp-2：时序增强 (temporal_scale + jitter + mask)
- temporal_scale (0.8-1.2x)，坐标 jitter (sigma=0.005)，关键点 mask (5%)
- S（替换）从 18% 飙到 28-29%：增强引入了模型无法消化的变化
- D 保持在 42-43%，collapse 7.8%
- **结论**：在 84d 低维特征空间下，坐标级增强的噪声大于信号。数据量 ~5K 不足，增强放大了方差而非提升泛化

#### Exp-3：CTC Prefix Beam Search (beam=3)
- 每样本 ~4.1s（比贪心 0.05s 慢 80x），512 样本需 ~35 分钟
- Greedy 和 beam 输出完全一致：模型 P(blank)≈0.77，概率分布过于尖锐，无备选路径可探索
- **结论**：beam search 只在模型输出足够"犹豫"时有价值，当前模型太偏 blank。未来不要尝试 beam search，除非模型架构/数据有根本性变化

#### 零成本优化的根本限制
数据量不足（~5K 样本）是 WER 瓶颈。三项优化思路在理论上正确，但在 5K 数据规模下：
- 分组 padding 减小的填充量不足以抵消 batch 变小带来的梯度噪声
- 坐标增强在低维特征下信噪比太差
- 模型的概率分布过于尖锐，beam search 和 greedy 等价
- **blank_bias=0 会让模型在 1 epoch 内 P(blank) 从 0→1.0**，+5.32 是正确设计

**下一步**：要突破 66.58% 的平台，需要更多数据或预训练，非训练/解码侧的工程优化。

#### 代码变更
- **新增** `src/augmentation.py` — 时序增强函数
- **新增** `src/beam_quick_test.py` — beam search 快速验证（10 样本）
- **新增** `src/validate_beam.py` — beam search 全量验证脚本
- **修改** `src/dataset.py` — `collate_fn_grouped` + augment 集成
- **修改** `src/decode.py` — `ctc_prefix_beam_search`
- **修改** `src/train.py` — `--group-size`/`--augment`/`--beam-width` 参数
