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
- **`src/model.py`**：1D Conv(84→256, stride=2×2 级联, 总降采样 /4) + BiLSTM(256→512, 2层,双向) + Linear(→vocab_size) + LogSoftmax + blank_bias=5.32，~13.5M 参数（3515 词表，84d kp-only；660d raw visual 时 ~14.0M）
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
- **文档校正**：ARCHITECTURE.md 训练流程描述（独立实现非复用 TFNet）、IMPLEMENTATION.md 词表 3515（非~500-1000）、模型 ~13.5M（非~5M）

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
4. **BiLSTM + CTC 天然易 blank 坍塌**：BiLSTM 在长 T/L 比下尤易塌陷（T/L=68:1），通过 1D Conv stride=2×2 级联（总 /4）+ blank_bias=+5.32 + blank penalty + 熵正则 + activity detection 联合解决

#### 尚待尝试（后续均已解决，见各日期条目）

- ~~stride=4 + 适度 blank bias（~-0.1，非 -4.0）~~ → 最终采用 **blank_bias=+5.32**（2026-07-31 验证必要性，见零成本优化节）
- ~~`zero_infinity=True` + stride=4（让 inf batch 跳过而非硬撑梯度爆炸）~~ → 训练改为 `zero_infinity=False` + 跳过 inf/nan batch
- ~~1D Conv+ResNet 替代当前模型（CSL-Daily 论文做法）~~ → 未采纳，模型架构不是瓶颈
- ~~先训最长句子（token 比空白多则 blank/T 比更低），而非全量数据~~ → 未采纳，用 activity detection + stride=2×2 级联解决
- ~~Conformer/Transformer encoder 替代 BiLSTM~~ → **未采纳**。实际模型保持 BiLSTM，配合 blank_bias + blank penalty + activity detection 解决坍塌（文档曾误标为 Conformer，09d3372 同步错误，2026-07-31 已更正）

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
- 均使用 blank penalty（threshold=0.55, weight=20.0）+ 熵正则（weight=0.01）+ 1D Conv stride=2×2 级联（总 /4）
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

目标：在不改模型架构、不加新数据的前提下，通过训练/解码优化将 WER 从 66.58% 降到 50-54%。

#### 实验矩阵

| 实验 | 配置 | Best WER | vs Baseline | 耗时 |
|------|------|----------|-------------|------|
| Baseline (best.pt) | bs=1, no augment, greedy | 66.58% | — | — |
| **Exp-1** | grouped bs=4 | **68.83%** | 退步 2.3pp | 66 epochs |
| **Exp-2** | bs=4 + temporal augment | **75.69%** | 退步 9.1pp | 89 epochs |
| **Exp-3** | beam search (beam=3) | 60.00%* | 持平* | 10 样本测试 |

*Exp-3 仅测试 10 样本，greedy 和 beam 结果完全一致。另有 `checkpoints/exp3_beam/` 一次从 best.pt 继续的 beam 验证训练（`--beam-width 3`），WER 84.44%→83.57% 无实质提升，同样支持"beam 无收益"结论。

#### Exp-1：Grouped Padding (bs=4)

- 预期 2-4pp 改善，实际退步 2.3pp
- 分组 padding 减少了填充量（~60%→~20%），但每组 B=2-4 样本，有效 batch 变小
- S（替换）从 18% 波动到 18%，D（删除）从 41.8% 波动到 47-53%，collapse 10% 左右
- 分组后 blank penalty 失效风险高，batch 小导致梯度噪声大
- **结论**：分组 padding 对 CTC 训练无帮助，序列长度差异本身是有效的正则化
- 注：`logs/exp1_bs4.log` 含两次训练（第一次 early stop e60 best=77.53%，第二次追加续跑 e66 best=68.83%）。`checkpoints/exp1_bs4/best.pt` 存的是第一次的 77.53%，68.83% 是第二次续跑的最优值（文档以 68.83% 计）

#### Exp-2：时序增强 (temporal_scale + jitter + mask)

- 预期 5-7pp 改善（这是零成本路线的主要收益来源），实际退步 9.1pp
- temporal_scale 改变序列长度（0.8-1.2x），坐标 jitter（σ=0.005），关键点 mask（5%）
- S 从 18% 飙到 28-29%：增强引入了模型无法消化的变化，token 预测更不准
- D 保持在 42-43%（与 Exp-1 相近），collapse 7.8%（低于 Exp-1 的 10%）
- **结论**：在 84d 低维特征空间下，坐标级增强的噪声大于信号。数据量 ~5K 不足，增强放大了方差而非提升泛化

#### Exp-3：CTC Prefix Beam Search (beam=3)

- 预期 5-8pp 改善，实际 0pp
- 每样本 ~4.1s（比贪心 0.05s 慢 80x），512 样本需 ~35 分钟
- Greedy 和 beam 输出完全一致：模型 P(blank)≈0.77，概率分布过于尖锐，无备选路径可探索
- **结论**：beam search 只在模型输出足够"犹豫"时有价值，当前模型太偏 blank，多路径探索不到替代 token。未来不要尝试 beam search，除非模型架构/数据有根本性变化

#### 根本原因

数据量不足（~5K 样本）是 WER 瓶颈。三项优化思路在理论上正确（分组 padding 减少填充浪费、增强提升泛化、beam search 替代贪心），但在 5K 数据规模下：
- 分组 padding 减小的填充量不足以抵消 batch 变小带来的梯度噪声
- 坐标增强在低维特征下信噪比太差
- 模型的概率分布过于尖锐，beam search 和 greedy 等价
- **blank_bias=0 会让模型在 1 epoch 内 P(blank) 从 0→1.0**，+5.32 是正确设计

### blank_bias=0 验证（为什么 +5.32 是正确初始化）

- 去掉 `fc.bias.data[0] = 5.32` 后训练：**P(blank) 在 1 epoch 内从 0% 冲到 ~100%**，blank 坍塌立刻复现
- 说明 +5.32（P(blank)=σ(5.32)≈0.3）不是 hack，而是必要初始化——给 blank 一个合理先验，既不让它占绝对优势，也不完全压死（对比 -4.0 时模型死锁，WER=100% 永远不动）

**下一步**：要突破 66.58% 的平台，需要更多数据或预训练模型，而非训练/解码侧的工程优化。

#### 代码变更

- **新增** `src/augmentation.py` — 时序增强函数
- **新增** `src/beam_quick_test.py` — beam search 快速验证（10 样本）
- **新增** `src/validate_beam.py` — beam search 全量验证脚本
- **修改** `src/dataset.py` — `collate_fn_grouped` + augment 集成
- **修改** `src/decode.py` — `ctc_prefix_beam_search`
- **修改** `src/train.py` — `--group-size`/`--augment`/`--beam-width` 参数
- 数据迁移：KEYPOINT_BASE → `E:/CE-CSL/CE-CSL`（CE-CSL 与 CSL-Daily 数据集从 D 盘迁移至 E 盘）
- 当前最优路线：全量 kp-only + activity detection（WER 66.85%），继续分析 token 级错误分布

### 2026-07-31 — 数据扩展双线并行（isolated words + L1290 YOLO）

#### 孤立词时序数据（agent: ac4ff5f）【完成】
- **新增 3 文件**：`src/extract_isolated_words.py`（MediaPipe IMAGE 模式 + normalize_hand 手腕归一化，输出 (T,84) float32）、`src/build_isolated_index.py`（clean_word(filename) 匹配 vocab_top478 word2idx）、`src/isolated_dataset.py`（IsolatedKeypointDataset + CombinedDataset 可混入 CE-CSL train）
- **归一化对齐**：E:/CE-CSL 主数据是手腕归一化（middle_mcp norm==1.0 已验证），孤立词复用同一公式，保证特征分布一致
- **中文路径**：cv2.VideoCapture 实测可直开真实中文文件名，仍内置 copy2→ASCII 临时目录回退防御
- **性能**：单进程 ~0.3 视频/s → 8 workers 复用 landmarker ~1.1 视频/s（每视频重载模型慢 10 倍）
- **最终产物**（主仓库 `isolated_words/`，gitignore 不跟踪）：basic 235 + common 863 = 1098 个 .npy（T 36-248，全部 dims={84} float32，0 错误）
- **index.json：覆盖 87 个唯一 token，94 个匹配视频**（basic 48 + common 46，7 个 token 跨两数据集）
- **验证**：CombinedDataset(CE-CSL 4910 + 孤立词 94) = 5004，DataLoader + collate_fn + SLRModel forward + CTC loss 全通过，activity_detect 可用（63→41 帧）

#### L1290 YOLO 静态手势分类（agent: ab5817c）
- **数据完整性检查**：train 2148 图/2149 标签（含 1 个多余的 `classes.txt`，无对应图，YOLO 自动忽略）、val 210/211（同理）。图片与标签一一对应，无空标签/无目标图，全部标签格式合法（`class cx cy w h` 归一化坐标 0-1），35 类均有样本，train 每类 37-108 样本、val 每类 4-11 样本。原图 640x480
- **任务**：l1290_data.yaml + train_l1290.py（yolov8s + pretrained）→ 训练 → 验证 mAP
- **重要坑（ultralytics 8.4.105）**：`YOLO("yolov8s.yaml").train(pretrained=True)` **不会真正加载 COCO 预训练权重**（yaml 构建的模型无 ckpt，bool pretrained 不触发 load_checkpoint，log 里无 "Transferred" 行）。必须传 `.pt` 文件：`YOLO("yolov8s.pt")` 才会加载。另外 GitHub 下载 SSL 证书验证失败（被墙/证书问题），需 `ssl.CERT_NONE` + urllib 手动下载 yolov8s.pt（22.6MB），AMP check 的 yolo26n.pt 下载同样失败但自动跳过不影响训练
- **训练配置**：yolov8s + COCO 预训练（Transferred 349/355 items），imgsz=640，batch=16（GPU 3.5-4.0GB），epochs=100，workers=0（Windows），optimizer=auto→AdamW(lr=0.000256)
- **进度（2026-07-31）**：训练运行中，~1.3 it/s，每 epoch ~100-110s + val
  - Epoch 1：mAP50=0.973，mAP50-95=0.742，P=0.895，R=0.878
  - Epoch 2：mAP50=0.983，mAP50-95=0.752，P=0.864，R=0.959
  - loss 快速下降：box 1.40→1.01，cls 4.58→1.35，dfl 1.50→1.15（前 2 epoch）
- **对比（弃用）**：首轮误用 yaml+pretrained=True 从随机初始化训了 7 epoch（box 1.43/cls 2.09/dfl 2.14，无 val），停掉改用正确预训练加载。随机初始化在小数据（2148 图）上收敛慢且 mAP 明显更差，COCO 预训练是关键
- **输出**：`YOLOv8/runs/l1290/`（results.csv、weights/best.pt、last.pt），随机初始化痕迹在 `runs/l1290_random_init/`（可删）
- **预计完成**：100 epochs × ~2min ≈ 3.5h（已跑 ~15min）

### 2026-07-31 — Agent D 归一化对照实验【完成】

#### 关键发现：数据本已归一化（任务前提错误）
- 全量 **1,247,783 个有效手帧**（train 1,049,611 + dev 95,249 + test 102,923）检查：**所有帧 wrist=[0,0]、wrist→MCP9 距离恒为 1.0**（0 例外）
- 66.58% baseline **本来就在已归一化数据上训练**。归一化在 Phase 1 后原地做掉（train_scales.json 记录归一化前尺度，.npy 文件时间戳晚于 scales）
- "归一化未实现"的 Landmine 是**文档幻觉**：E: 数据已归一化，但 main 分支 `src/preprocess_keypoints.py` 仍是原始坐标版（归一化代码在 `worktree-fix-preprocess-normalization` 分支）

#### 交付代码（已合入）
- `src/preprocess_keypoints.py`：新增 `normalize_hand()`（wrist 原点 + 除 MCP9 距离，epsilon=0.1 防除零）
- `src/normalize_existing.py`：离线复归一化（8 进程，5987 文件 ~10s）
- `src/train.py` + `src/dataset.py`：`--keypoint-subdir` / `--keypoint-dir` / `--epochs` 参数

#### 15-epoch 对照实验（两个 Agent D 实例交叉验证）
| Epoch | D1 | D2 |
|-------|----|----|
| 5 | 96.52% | 94.42% |
| 10 | 89.36% | 90.17% |
| 15 | 85.36% | 84.90% |
| best | 85.36% (e15) | 83.88% (e12) |

- 两实例结果**交叉一致**（±1.5pp），训练轨迹 = baseline 冷启动复跑（blank 坍塌→恢复，仍在下降）
- **结论**：归一化是 baseline 已有状态，**不是 WER 瓶颈**。对已归一化数据再归一化不可能改变 WER。真实 raw-vs-normalized 对比无法做（原始坐标已不存在）
- 与既有结论一致：**瓶颈是数据量（~5K）**，下一步数据扩展（孤立词 94 已交付 + CSL-Daily 20K）

### 2026-08-01 — L1290 YOLO 训练完成【完成】

- **训练配置**：yolov8s.pt（COCO 预训练）+ l1290_data.yaml，imgsz=640，batch=16，epochs=100，workers=0，AdamW(lr 自适应)
- **训练中途被外部杀死一次**（epoch 35，无 traceback，疑似父 shell 清理）。用 `train_l1290.py --resume`（新增 flag，从 last.pt 恢复）以 PowerShell Start-Process 脱离会话重启，从 epoch 35 续跑到 100
- **最终指标**（val 210 图）：
  - 最终 epoch 100：P=0.982，R=0.985，**mAP50=0.983**，mAP50-95=0.795
  - **最优 mAP50=0.985**（epoch 18），最优 mAP50-95=0.805（epoch 55）
  - epoch ~20 后 mAP 即达饱和（0.98+），后续为损失微调
- **结论**：L1290 35 类静态手势分类在 COCO 预训练下 100 epoch 收敛良好，作为旁路 YOLO 静态手势识别的权重候选（`YOLOv8/runs/l1290/weights/best.pt`）
- **顺带**：`src/preprocess_keypoints.py` 合入 Agent D 的 `normalize_hand()`（wrist 原点 + MCP9 距离归一化），与 E: 已归一化数据公式一致，经功能测试（wrist→origin、MCP9→distance 1.0、退化尺度回退 0.1）

## 2026-08-01 — 全实验总结：数据量（~5K）是 WER 瓶颈的唯一结论

8 条 agent 线全部完成（A 视觉 ×2、C 文档 ×2、D 归一化 ×2、孤立词、L1290）。全部受控实验均指向同一结论：

| 实验线 | 假设 | 结果 | 结论 |
|--------|------|------|------|
| Baseline | 当前最优 | WER 66.58% | — |
| 零成本优化 ×3 | 分组 padding / 时序增强 / beam search | 68.83% / 75.69% / 0pp | 均退步或不改善 |
| Visual features | MobileNetV3 特征提升识别 | 84.79% / 91.61% | ImageNet 特征与手语无关，非瓶颈 |
| Agent D 归一化 | "归一化未实现" | 数据早已归一化，复归化无效 | 归一化不是瓶颈 |
| Isolated words | 数据扩展（94 孤立词混入） | 代码交付，未重训 | 数据量路线 |
| **L1290 YOLO** | 旁路静态手势分类 | **mAP50=0.985** | 旁路就绪 |

**唯一结论**：WER 66.58% 的天花板来自数据量（~5K 连续句样本），而非模型架构、特征维度、解码策略或归一化。下一步突破只能靠数据扩展（CSL-Daily ~20K 或 SLR 25K 孤立词预训练）。

## 2026-08-01 — 读 CE-CSL 原论文（2409.11960）+ TFNet 轻量化蒸馏路线

### 合并 + 双远端推送【完成】
- main fast-forward 到 f1b3b2e（全部实验 + 文档交叉验证内容），推送 origin（github）+ gitee2 双远端成功（c4c8380..f1b3b2e）

### 论文要点（哈工程 CE-CSL 原论文，TFNet 即 WER.py 来源项目）
- 数据集：12 表演者（A-L，8 女 4 男，I/J 听障），5988 视频，3515 词，10.52h，70+ 生活场景，train/dev/test = 4973/515/500，OOV=0
- **TFNet 在 CE-CSL 上 Dev/Test WER 42.1%/41.9%（本文 SOTA，第二名 MAM-FSD 44.9%）**；RWTH 18.6%/18.6%、RWTH-T 18.0%/19.1%、CSL-Daily 25.1%/23.5%
- 官方 README 称最新权重 test WER **32.46%**，在百度网盘（提取码 0000）
- TFNet 架构：RGB 全帧端到端（MAM-FSD CNN backbone）→ 双分支（时域 1D CNN+BiLSTM ⊕ 频域 DFT+1D CNN+BiLSTM）→ 相加 → FC；Loss = CTC + 双 VAE 辅助损失
- 训练细节：Adam lr=1e-4，**batch_size=2**（与我们一致），55 epochs，35/45 epoch lr 降 80%；随机裁剪 256→224 + 水平翻转(0.5) + 时序 ±20% 缩放；解码 beam=10
- 消融：纯 CTC 21.3% → +双 VAE 18.6%（RWTH 上 -2.7pp）；时序+频域融合比纯时域 -0.8pp
- 标注规范（佐证 clean_word 去括号逻辑）：方向词带括号（`他/帮(我)/开门`）、方言词有 Note 注记、数字词自动补齐

### 轻量化评估：TFNet 方案不满足轻量需求
- TFNet 推理 = 每帧全 RGB 过 CNN（224×224）+ 两套 BiLSTM（时域+频域）+ beam=10，吃 GPU；正是本项目"低算力退化为纯时序识别"想省掉的成本
- 结论：直接搬正面违反 v1 约束。但双 VAE 辅助损失是纯训练时正则（推理零成本，RWTH 上 -2.7pp），可迁移；频域分支收益小（-0.8pp）要养一套 BiLSTM，性价比低不迁

### 蒸馏路线：重训练一次，轻推理永远【已记入 INSPIRATION #4】
- **机制**：TFNet（RGB teacher，42%/32.46%）→ 蒸馏 → 我们的关键点模型（student，轻推理不变）
- 训练期重：RGB 帧与关键点帧**序号天然对齐**，teacher CTC 软概率做 frame-level KL；部署期轻：仍是 84d 关键点 + 单 BiLSTM + 贪心，MediaPipe ~63ms/帧
- loss：`α·CTC(student,GT) + β·KL(student,teacher)` 软标签；或**伪标签双杀**——teacher 跑 SLR 25K / CSL-Daily 20K 无标注数据生成 gloss，顺带解决数据扩展
- **成本**：teacher 一次训练最贵（RTX 3090Ti 24GB，训练量 ~10-20x 当前）。本地 TFNet-main 无 checkpoint（仅源码+百度网盘地址），可用官方权重当 teacher 免自训
- **预期上限**：~24pp 差距大头是表征鸿沟，蒸馏只能教对学生能表达的东西（对齐/切分/去误插），落点 ~60% 出头，别期待 40%
- 作者团队蒸馏是成熟路线：2303.06820 cross-resolution KD（本地 kdloss 出处）、2402.19118 frame-level 自蒸馏、2207.00928 TSRNet
- **待定**：teacher 词表 3515 vs student 子词表 479 的软标签对齐方式（student 升 3515 或 teacher 聚到子词表），动手时再定

## 2026-08-01 — CSL-Daily 关键点接入 + 混合训练启动【进行中】

### 四方向 agent 汇总（顺序执行：3 → 4 → 1 → 2）
- **方向4 双 VAE 辅助损失【已实现】**：`src/model.py` 加 `SequenceVAE`（接 BiLSTM 输出逐帧重建 + KL），`src/train.py` 加 `--vae-loss/--vae-weight`（默认关，weight=0.01）。CPU 冒烟通过，默认行为逐位不变。VAE loss 量级 ~62 vs CTC ~3.5，weight 0.01 缩放后同量级
- **方向1 TFNet teacher 权重【已分析】**：879MB checkpoint（epoch=38, dev 33.19，即 CSL-Daily→CE-CSL 微调版）。ResNet34MAM 双分支（时域 + FFT 频域）→ 2 层 BiLSTM(1024) → NormLinear(1024,3516=3515+blank)。词表与我们 99.9% 对齐（交集 3513，差标点+`２`）。加载：strict=True 全绿（22/44 分类头是别名副本需补进 sd）。**硬约束：单视频前向 ~830ms/1.46GB，只能离线缓存软标签**。90 帧窗口 teacher T'=23 vs 学生 T'=22 需插值。top-478 投影需边缘化 3037 词
- **方向2 SLR 孤立词【已分析】**：`slr500_words_joints/` 是 125K 个 .body.txt（Kinect 全身 25 关节，与 84d 不兼容，**废弃**）。真 npy 在 `SLR_Dataset/keypoints/train/`：**24,770 个 (T,84)** 与 CE-CSL 逐点吻合，但**只覆盖类 000-110（111/500 词），抽取中断**，全量应 11 万。词表命中 top478 仅 14%。**暂缓**（需续抽 389 类 + 命中低）
- **方向3 CSL-Daily【已接入】**：`E:/CSL-Daily/CSL-Daily/keypoints/` 20,653 个 (T,84) **是本项目 MediaPipe 管线抽的**（非 Kinect），与 CE-CSL 逐字节一致，frames_512x512 也完整。新增 `src/csldaily_dataset.py`（读 keypoints/{split} + labels.json，返回与 KeypointDataset 一致 dict，可复用 collate_fn/CombinedDataset）

### 词表命中率分析（决定用全量 3515 而非 top-478）
| 词表 | CSL-Daily token 命中 | 样本数 |
|------|---------------------|--------|
| top-478 子词表 | **65.6%** ❌ 唯一 token 命中仅 19.7% | 18315 |
| **全量 3515** | **90.5%** ✅（唯一 token 66.8%） | ~18K |

top-478 下 CSL-Daily 大量标签被静默丢弃（`_gloss_to_ids` 容错跳过），信号稀释严重；全量 3515 下命中 90.5%，且与 teacher 词表对齐（为蒸馏铺路）。**决定：混合训练用全量 3515 词表**。

### train.py 接入
- `--csldaily-base` 参数：非 None 时 CSL-Daily train 混入训练集（CombinedDataset），dev 保持 CE-CSL
- 修复 `epochs = 100` 硬编码 → `args.epochs`（VAE agent 改动时遗漏，实际 `--epochs` 一直未生效）
- 冒烟测试：CE-CSL 4910 + CSL-Daily 18315 = 23225 样本，DataLoader+collate+前向+CTC loss 全通过

### 训练启动【进行中】
- 命令：`train.py --vocab vocab.json --csldaily-base E:/CSL-Daily/CSL-Daily --activity-detect --vae-loss --visual-fusion none --checkpoint-dir checkpoints/csldaily_vae`
- detached PowerShell 启动（PID 32732），日志 `logs/train_csldaily_vae.log`
- Epoch 1 实测 ~27 it/s，11686 batch/epoch ≈ 7 分钟，100 epochs ≈ 11-12h
- 假设：数据量 4.7 倍应显著降 WER（验证"数据量是唯一瓶颈"）；VAE loss 顺带验证论文 -2.7pp 是否复现

### 2026-08-01 夜 — 三路诊断：4.7 倍数据为何收效甚微【Epoch 70, best 66.76%】

**现象**：混合训练到 Epoch ~69 best WER 66.76%，与 baseline 66.58% 几乎持平（collapse 31.2%→2.7% 大幅改善，但 D 41.8%→46.4% 退步）。4.7 倍样本几乎没换来 WER 下降。

**三路 CPU 诊断 agent 结论（不占 GPU）**：

1. **域偏移（A3）**：两数据集坐标量纲、运动能量完全一致，**唯一显著偏移是缺手率 CE 30.1% vs CSL-Daily 2.55%（12x）**。CSL-Daily 干净帧主导梯度，稀释 CE 学"无手帧→blank"的能力 → 解释了 D 退步。帧长反向（CE 183 vs DL 120）。**可管理偏移**，建议混入比例下调或对 CE 缺手帧加权
2. **OOV 映射（A1）**：CSL-Daily 663 个 OOV 词占 9.5% token，被 `_gloss_to_ids` 静默丢弃成负信号。分类：A 可分解 21.5%（组合词如"好了"）、B 近匹配仅 1.0%、**C 真新词 77.4%**（516 词，count≥10 有 297 词可并入词表回收 61% token，3515→3812 仅 +2.25% 参数，但样本稀疏易欠训）
3. **低频词覆盖（A2）**：dev 250 个 CSL-Daily 未覆盖瓶颈词（占 22.4% token），其中 132 个 train 出现 ≤3 次。数据源覆盖：**isolated_words 数据全在但索引不全**（见下）、SLR 已提取 190 类补 9 词、SLR 定向 10 类补 10 词

**关键发现：isolated_words 索引缺口**——`isolated_words/keypoints/` 实际有 **1057 词 npy**（basic 235 + common 863 全量，Phase 5 已全提取），但 `index.json` 只收录 **87 token**，**167 个在词表内的词从未被训练使用**（含 dev 8 个 ≤3 次最难词：信任/停车/圣诞节/挫折/明年/星期六/星期日/疼）。

### 2026-08-01 夜 — E3 孤立词数据解锁【完成】

- **重建 `isolated_words/index.json`**：87 → **254 token，271 样本**（167 新增），质量验证通过（帧长 36-139 全正常），无旧 token 丢失
- 覆盖 dev 瓶颈词 **23/235**（含 8 个 ≤3 次最难词），另 75 个词与 CSL-Daily 覆盖重合
- **`src/train.py` 加 `--isolated-index` 参数**：非 None 时 `IsolatedKeypointDataset` 混入训练集（CombinedDataset），冒烟测试 CE-CSL 4972 + CSL-Daily 18400 + isolated 271 = 23643 样本通过
- 提交 08d73ca（worktree 分支）
- **下一步（GPU 实验清单）**：E1 OOV 词表扩展重训（3515→3812 并 C 类 297 词）、E2 混入比例下调（缺手率域偏移修正）、E3 已完成、E4 SLR 定向 10 类（公务员/天文/就业/按钮/眼 等 5 个 ≤3 次词）

### 2026-08-02 — 新训（isolated 混合）突破 baseline【进行中】

**训练状态**：`checkpoints/csldaily_iso/`，命令 `train.py --vocab vocab.json --csldaily-base E:/CSL-Daily/CSL-Daily --activity-detect --vae-loss --visual-fusion none --isolated-index "D:/red star project/isolated_words/index.json" --checkpoint-dir checkpoints/csldaily_iso`（PID 50848）

- 数据组成：CE-CSL 4972 + CSL-Daily 18400 + 孤立词 271 = **23643 样本**
- **best WER 64.24%（Epoch 77/81，S=20.1% D=42.4% I=1.7%）**，跌破 66.58% baseline **1.81pp**，较旧训 66.76% 提升 2.52pp
- 关键轨迹：Epoch 47 平台 67.45% → lr 降至 0.000125 后 Epoch 63 首次跌破 baseline（65.25%）→ Epoch 77/81 至 64.24% → Epoch 84 lr 再降至 0.000063 后 64.36%
- **验证了"词汇覆盖是瓶颈"假设**：孤立词补的是 S（分类能力 21.3→20.1%），D（缺失检测）回到 41.8% 与 baseline 持平。CSL-Daily 混合单独用无效，必须配孤立词解锁
- **代码审查发现长度对齐 bug**（Workflow 确认中）：model.py:97 `input_lengths=(L//4)` vs conv 实际 `ceil(L/4)`，L%4≠0 的 42 个长度下每序列末帧被静默丢弃（~4.5% 监督帧），严重度低-中，非 22pp 差距主因

### 2026-08-02 — Web 化前后端识别系统【规划，待实现】

**动机**：Phase 5 YOLO 旁路权重已训好（L1290 mAP50=0.985），但主路实时管线（Phase 4）未实现。为验证训练结果 + 承载主路/静态词，搭建 Web 化实时识别系统。架构详见 `ARCHITECTURE.md` v2 章节。

**分阶段**：
- **S1（本次样品）**：只接旁路 YOLO（帧→YOLO→35类→防抖→前端显示），验证 `l1290 best.pt` 实际效果。半天跑通
- **S2**：接主路（MediaPipe→84d→滑动窗口→BiLSTM+CTC→句子），顺带完成 Phase 4 实时管线
- **S3**：仲裁层融合双路 + config.yaml（解决路径硬编码待办）

**技术选型**：FastAPI（原生 asyncio+WS）后端 + 浏览器 getUserMedia 摄像头 + WS 传 JPEG 帧 @10fps + 推理独立线程。

**S1 文件布局**：`backend/main.py`（FastAPI+WS）、`inference/static_yolo.py`（YOLO 封装）、`frontend/index.html + app.js`。
**依赖**：fastapi / uvicorn / websockets 需新增（当前 yolov8 环境未装），补 requirements.txt。
**权重路径**：YOLO 权重实际在 worktree `.../exp+zero-cost-optimization/YOLOv8/runs/l1290/weights/best.pt`（主仓库该路径不存在，CLAUDE.md 已注明）。
**待确认**：摄像头在浏览器端（推荐）还是后端 cv2 直读；静态词先用 35 类还是等 SLR 完整词表。

### 2026-08-02 — S1 Web 系统落地【完成 + 平板适配进行中】

#### S1 前后端搭建【完成】

按"前端 UI agent + 后端推理 agent"分工（用户要求多 agent 并行）搭建完成，端到端验证通过：

**后端**（`backend/main.py` + `inference/static_yolo.py`）：
- **确认 L1290 是 yolov8s 检测模型**（task=detect，35 类，非分类），`infer()` 取整帧最高置信度检测框 class_id → 类名（从 `YOLOv8/l1290_data.yaml` names 读，如"时间/时候/你/您"）
- 防抖：同类别连续 ≥15 帧 + 置信度 ≥0.5 才输出词；WS 无帧超时 2s 重置防抖状态
- 推理跑独立 daemon 线程（queue.Queue 通道），WS 事件循环不阻塞（修复了 `queue.get` 直接放协程导致 WS 握手超时的 bug）
- `/health` 返回 `{"status":"ok","model_loaded":true,"classes":35}`，单帧推理 ~17ms GPU
- **注意**：后端 agent 验证时误用 `taskkill //F //IM python.exe` 按进程名全杀，连带杀掉了 CSL-Daily 训练进程（PID 50848，Epoch 87/100 中断，best 64.24% 已存档）。**教训：停服务必须用精确 PID，禁止按进程名全杀**（详见下方 Landmine）

**前端**（`frontend/index.html` + `app.js`）：getUserMedia 采集 → Canvas 480p + JPEG q0.6 → 10fps WS 上行；深色主题、中文 UI、词卡片（最近 5 个 + 置信度 + 时间戳）、状态灯（连接/识别/置信度）、丢帧统计（用 ws.bufferedAmount 积压估算，因后端无逐帧 ack）

**依赖**：requirements.txt 追加 `fastapi>=0.140`、`uvicorn>=0.52`、`websockets>=16.0`（已装入 yolov8 环境：0.141.1/0.52.1/16.1.1）

**验证**：`/health` 通过；WS 端到端收到 `{"type":"static_word","word":"时间/时候","conf":0.923}`；GET / 返回 index.html。

#### 训练进程被误杀【事故 + 决策待定】

- S1 验证重启服务器时 `taskkill //F //IM python.exe` 全杀，误杀 CSL-Daily 混合训练（PID 50848）
- 中断于 Epoch 87/100，`best.pt` = **64.24%**（Epoch 81 存档，完整）。Epoch 87 本身 wer=64.24% 平 best 且 **D=40.3% 历史最低**，说明还有下降空间
- train.py 无 `--resume`，续跑需从头（~14h/100epoch）。等待用户决策：接受 64.24% / 修复 bug 后重训 / 等代码审查出结果再定

#### 平板适配【进行中，多 agent 分工】

用户要**平板直接测试**。核心障碍：getUserMedia 要求 secure context，平板经 `http://192.168.1.8:8000`（局域网 WLAN IP）访问会被浏览器拦截摄像头 → **必须 HTTPS**（自签证书，平板接受警告后可用）。

Workflow（w7p1lwu5x，3 agent 分工）：
- 后端 agent：生成自签证书 `certs/`，main() 检测证书存在即 HTTPS + 绑定 0.0.0.0，写 start_server.py 一键启动，停旧服务（精确 PID 13396）后重启 HTTPS
- 前端 agent：平板触屏适配（大按钮/大字号、@media 响应式横竖屏、https 提示条、全屏按钮）
- 验证 agent：curl -k 验证 /health + wss 推帧端到端

前端已改：`app.js` WS URL 自适应（页面 https → `wss://location.host`，否则 `ws://`）。

**平板访问路径**：平板浏览器打开 `https://192.168.1.8:8000` → 接受证书警告 → 启动摄像头即测。
