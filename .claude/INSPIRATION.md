# 启发日志

从其他项目/论文/代码库中发现的架构模式、工程技巧、优化思路，暂未纳入实现，供后续讨论选型参考。

---

## 来源：TFNet-main（已读 2026-07-29）

### 1. TemporalConv + 长度追踪
- **文件**：`TFNet-main/Module.py:25-70`
- **内容**：Conv1d 时序下采样（K5+P2+K5+P2），90 帧 → ~20 帧再送 BiLSTM
- **好处**：推理更快（输入短 4 倍）；`update_lgt()` 每次下采样自动更新有效长度
- **代价**：~30 行代码，模型需加一层 TemporalConv
- **适用场景**：v1 即可用，训练前加即可
- **状态**：待讨论

### 2. collate_fn stride-aware pad
- **文件**：`TFNet-main/DataProcessMoudle.py:435-473`
- **内容**：pad 到 `total_stride` 整数倍 + 边界余量，避免卷积边界伪影
- **好处**：长度追踪对齐——加入时序卷积后 `input_lengths` 自动匹配
- **代价**：约 20 行，改 collate_fn
- **适用场景**：配合第 1 点一起用
- **状态**：待讨论

### 3. TemporalRescale 数据增强
- **文件**：`TFNet-main/videoAugmentation.py:219-241`
- **内容**：随机 ±20% 时间缩放（帧数变长/变短），模拟不同手语速度
- **好处**：CTC 训练对这种时序变化敏感，对 .npy 做插值比原始视频快得多
- **代价**：~10 行，Dataset 里加一个 transform
- **适用场景**：v1 即可用
- **状态**：待讨论

### 4. SeqKD 蒸馏（v2）
- **文件**：`TFNet-main/DataProcessMoudle.py:521-538`
- **内容**：不同时间分辨率输出互相 KL 蒸馏，免费模型集成效果
- **好处**：不增加推理成本，通常提升 2-5% WER
- **代价**：~15 行，需双分支模型结构
- **适用场景**：v1 效果不够好时
- **状态**：v2 考虑

### 5. WERAugment（v2）
- **文件**：`TFNet-main/videoAugmentation.py:25-81`
- **内容**：gloss 级别做删除/替换/插入，模拟手语漏词/重复
- **好处**：数据增强，提升泛化
- **代价**：需要 gloss 时间边界标注（我们没有）
- **适用场景**：v2 若有边界标注数据
- **状态**：v2 考虑，数据缺失

### 6. NormLinear 分类头
- **文件**：`TFNet-main/Module.py:72-80`
- **内容**：L2 归一化权重做分类 `F.normalize(weight, dim=0)`，防大类压制
- **好处**：手语词频极度不均，NormLinear 对长尾词友好
- **代价**：~5 行，替换 `nn.Linear`
- **适用场景**：v1 即可用，替换 model.py 最后一层
- **状态**：待讨论

---

## 来源：reference/ 参考项目（已读 2026-07-29）

### wenet — 流式 ASR 生产框架

#### 9. Strided Chunk 流式编码 + Cache 传递
- **文件**：`wenet/models/transformer/encoder.py:302-362` (`forward_chunk_by_chunk`)
- **内容**：训练时用 chunk mask 模拟流式，推理时按 `stride = subsampling_rate × chunk_size` 滑动，attention KV cache + CNN cache 跨 chunk 传递避免重复计算
- **好处**：全流式低延迟（chunk_size × 40ms 以内），与离线训练共享权重
- **代价**：需单向 LSTM（当前 BiLSTM 不兼容），需管理额外 cache 张量
- **适用场景**：v1 可直接借鉴——滑动窗口每新帧到达时，LSTM hidden state 跨窗口传递，不从头跑 90 帧
- **状态**：待讨论

#### 10. CTC Prefix Score 增量维护（无 LM beam search）
- **文件**：`wenet/models/transformer/search.py:64-101` (PrefixScore)，`espnet2/legacy/nets/ctc_prefix_score.py:12-361`
- **内容**：每前缀维护 4 个分数（blank/non-blank × viterbi/forward），新帧到达时按 CTC 折叠规则递推更新 `log_add`。每帧 O(beam × vocab)，不依赖 attention decoder 或 LM
- **好处**：比贪心解码显著提升质量（尤其重复/缺失检测），计算开销可控。**wenet 和 espnet 对这一模式高度一致**
- **代价**：内存随 beam 线性增长，需显式管理前缀字典
- **适用场景**：v2 直接替换贪心解码，新帧到来时增量更新 next_hyps（不从头扫描）
- **状态**：v2 强烈推荐

#### 11. 流式管线三层架构（Feature → Encoder → Decoder）
- **文件**：`wenet/models/transformer/asr_model.py:216-343`，`runtime/gpu/client/speech_client.py:62-147`
- **内容**：管线组织为 Feature Extractor(chunk 级生成) → Encoder(forward_chunk_by_chunk) → Decoder(CTC log_softmax)。用 `sequence_id` 跨 chunk 追踪，`simulate_streaming` 标志控制编码器路径
- **好处**：离线训练/非流式推理/流式推理共用同一权重，模块解耦清晰
- **代价**：需在 Phase 4 推理入口维护 session 状态字典
- **适用场景**：v1/v2 均可参考——MediaPipe 提取 → 帧窗口化 → LSTM(chunk 增量) → CTC 解码
- **状态**：待讨论

#### 12. 多路并发状态管理（Session State Pattern）
- **文件**：`wenet/runtime/gpu/model_repo_stateful/wenet/1/model.py:72-180` (TritonPythonModel)
- **内容**：用 `self.seq_states[corrid]` 存储每路序列完整状态（CTC 前缀树 + encoder_out 历史）。START 初始化 → chunk 到达时取出/推进/更新 → END 触发 rescoring 后清理
- **好处**：多用户并发流式推理无状态泄漏，天然支持多摄像头场景
- **代价**：需服务端长连接 + 内存管理，v1 单用户不急需
- **适用场景**：v2 多路并发时标准解法
- **状态**：v2 保留

#### 13. Attention Rescoring（CTC + Attention 融合重评分）
- **文件**：`wenet/models/transformer/search.py:374-458` (`attention_rescoring`)
- **内容**：CTC prefix beam search 产生 nbest + 时间戳 → attention decoder 逐 token 重评分 → 最终 = decoder + ctc_weight × ctc + reverse_weight × r_decoder。重评分仅在 utterance 结束时触发一次
- **好处**：CTC 提供时间对齐 + token 级 attention 提供上下文感知，效果显著优于纯 CTC
- **代价**：需额外训练 attention decoder，重评分一次性全序列操作（非逐帧）
- **适用场景**：v2（v1 CTC 准确率不够时可以考虑——在静默检测后触发重评分）
- **状态**：v2 保留

### espnet — 流式 ASR 研究框架

#### 14. Config-Driven YAML 架构
- **文件**：`espnet/egs2/aishell/asr1/conf/train_asr_streaming_conformer.yaml`
- **内容**：模型结构、训练参数、解码参数全由 YAML 驱动。`encoder`/`decoder` 字段指定模块类型，`*_conf` 传递 kwargs。`ASRTask.build_model_from_file()` 统一实例化
- **好处**：零代码切换模型变体，实验迭代极快
- **代价**：初始写 YAML schema ~50 行
- **适用场景**：v1 即可引入——用 YAML 管理滑动窗口参数、模型维度、训练超参（已在 CLAUDE.md 待办中）
- **状态**：待讨论

### landmark-extraction — 手语关键点提取工具链

#### 15. NaN 替代零值作为缺失标志
- **文件**：`landmark-extraction/src/extractor.py:134-153, 315-326`
- **内容**：未检测到手部时用 `np.full(shape, np.nan)` 而非填零。NaN 语义清晰表达"无数据"，不污染下游坐标分布，下游用掩码或插值处理
- **好处**：零计算开销，消除零值歧义（真实坐标 (0,0) vs 缺失）
- **代价**：下游模型/tensor 操作需处理 NaN（PyTorch 支持 masked operations）
- **适用场景**：v1 可直接采用——替换当前 `visibility < 0.6 → 置零` 为 `→ NaN`
- **状态**：待讨论

#### 16. 固定双手张量结构 [T, 2, 21, 3]
- **文件**：`landmark-extraction/src/extractor.py:319-326`
- **内容**：每帧始终输出 `[T, 2, 21, 3]`（手×点×xyz），未检测到的手槽填 NaN。下游输入形状完全固定，无需动态折叠
- **好处**：消除变长输入复杂性，简化 batch 和 padding
- **代价**：存储约 2 倍（两手各存一份，不管是否检测到）
- **适用场景**：v1 可考虑——当前 (T, 84) 扁平表示混在一起，分两侧更有语义
- **状态**：待讨论

#### 17. 数据集 NaN% 缺失率分析
- **文件**：`landmark-extraction/src/analyze.py:49-52, 96-101`
- **内容**：按 signer/词类/视频统计 landmark NaN 比例，生成热力图。迅速发现哪些词类或 signer 检测质量差，指导数据清洗
- **好处**：一次性脚本 ~50 行，避免"脏数据进模型"的隐藏问题
- **代价**：训练前多一步分析
- **适用场景**：v1 训练前——对所有 .npy 跑一遍，确认数据质量合格
- **状态**：待讨论

### slt — 手语翻译专用项目

#### 18. Gloss 后处理清洗流水线
- **文件**：`slt-master/signjoey/phoenix_utils/phoenix_cleanup.py`
- **内容**：正则规则清洗 CTC 解码输出——移除标注噪声标记（`__LEFTHAND__` 等）、合并多词复合 gloss、去连续重复。手语识别特有的后处理，弥补 CTC 贪心解码碎片化
- **好处**：零计算开销，可提升 WER 2-5%
- **代价**：需针对 CSL 标注规范定制正则规则集
- **适用场景**：v1 直接可用——在 `src/decode.py` 后处理中加规则清洗
- **状态**：待讨论

#### 19. 肩距归一化（替代手腕间距）
- **文件**：`slt_how2sign/fairseq/data/sign_language/sign_features_dataset.py` (NormType.body)
- **内容**：以左右肩距离为参考尺度做坐标归一化 `pose.normalize(normalize_info)`，消除体型和相机距离差异。支持 kp_wise/global_xyz 三种模式
- **好处**：预处理阶段完成，训练零开销；肩距比手腕间距更稳定（手腕快速运动时波动大）
- **代价**：MediaPipe Hands 不含肩膀点，需改用 Holistic（违反 v1 约束）
- **适用场景**：v2 若引入 Holistic 可用；v1 维持手腕间距
- **状态**：v2 保留（与 v1 "不做 Holistic"约束冲突）

#### 20. Reduced BLEU — 手语翻译专用评估
- **文件**：`slt_how2sign/fairseq/tasks/sign_to_text.py` (reducedBLEU/reducedchrf)
- **内容**：计算 BLEU 时先过滤黑名单高频虚词（英文 the/a/is，中文的/了/是/在）。手语表达省略虚词，标准 BLEU 过度惩罚这种省略
- **好处**：仅评估阶段使用，零训练开销；更符合手语翻译评价直觉
- **代价**：需维护中文虚词黑名单
- **适用场景**：v1/v2 均可——作为 WER 的补充评估指标
- **状态**：待讨论

#### 21. 联合识别+翻译多任务学习
- **文件**：`slt-master/signjoey/model.py` (SignModel.forward)
- **内容**：共享编码器 + CTC 头（gloss 识别）+ Decoder 头（口语翻译）。`w_rec × CTC_loss + w_trans × XEnt_loss` 联合优化，gloss 作为中间监督改善翻译质量
- **好处**：翻译 BLEU 可提升 3-5%，天然同时输出 gloss 和译文
- **代价**：需 gloss 标注数据，训练开销约 +20%
- **适用场景**：v2（v1 无 gloss 边界标注数据）
- **状态**：v2 保留

### k2 — FSA/FST 语音识别框架

#### 22. CTC 拓扑 FSA Compose（训练与解码统一构图）
- **文件**：`k2/python/k2/fsa_algo.py:1252-1281` (ctc_topo)，`k2/python/k2/ctc_loss.py:62-98`
- **内容**：用 FSA compose 构建解码图（CTC 拓扑 FSA ∘ 转录 FSA），`aux_labels` 为输出 token。换 lexicon、加 LM 只需替换 compose 输入。DenseFsaVec 将 NN log_softmax 包装为 FSA，训练/解码在同一构图框架下进行
- **好处**：极强解耦——换 lexicon、加 LM、换 loss 均零改动核心训练/解码逻辑。`delay_penalty` 可在训练阶段直接注入
- **代价**：抽象层较高，调试不如直接操作 tensor 直观；需理解 FSA compose 机制
- **适用场景**：v2 引入 LM 或需要训练阶段注入流式延迟约束时
- **状态**：v2 保留

#### 23. OnlineDenseIntersecter — 帧同步 Viterbi 解码状态机
- **文件**：`k2/python/k2/online_dense_intersecter.py:29-141`
- **内容**：每流维护 `DecodeStateInfo`，逐帧/逐 chunk 调用 `decode(dense_fsas, decode_states)` 做 intersect → 局部 lattice → 剪枝（search_beam/output_beam/min_active_states/max_active_states）。状态跨帧传递，多流并行
- **好处**：天然适配实时场景，beam=1 退化即为我们当前的贪心解码
- **代价**：需 C++ 后端支持（但逻辑思想可直接参考）
- **适用场景**：v1/v2 思想层面直接可用——当前贪心解码可视为此方案 beam=1 特例
- **状态**：参考

---

## 来源：MedSight（已读 2026-07-29）

### 7. 统一模块接口（BaseAgent 模式）
- **文件**：`med-sight-main/src/agents/base_agent.py`
- **内容**：所有 agent 同一生命周期 `validate → pre_process → process → post_process → metrics`
- **好处**：6 个推理模块都返回 `ModuleResult(success, data, confidence, error)`，联调时一眼看出哪环断
- **代价**：给每个模块包一层 ~10 行
- **适用场景**：Phase 4 推理管线
- **状态**：待讨论

### 8. 结构化结果（AgentResult 带置信度）
- **文件**：`med-sight-main/src/agents/base_agent.py:34-65`
- **内容**：每个 agent 返回带 `success`/`confidence`/`error`/`metadata` 的结果对象
- **好处**：仲裁层可直接用各模块置信度，不依赖内部实现细节
- **代价**：定义 dataclass ~15 行
- **适用场景**：Phase 4 推理管线
- **状态**：待讨论

---

## 头脑风暴 2026-07-29

> 5 位专家 agent（wenet、espnet、landmark-extraction、slt、k2）交叉评审全部 23 条启发，
> 从各自专长领域给出采纳建议、冲突分析和组合方案。以下是汇总。

### 投票结果：v1 最该立刻采纳

| 票数 | 条目 | 投票者 | 理由 |
|------|------|--------|------|
| ★★★ | **#15 NaN 替代零值** | espnet, landmark, slt | 5 行代码，消除零值歧义，不影响模型 |
| ★★★ | **#17 NaN% 缺失率分析** | espnet, landmark, k2 | 1h 脚本，训练前知道数据质量 |
| ★★ | **#14 Config-Driven YAML** | espnet, wenet | 零模型改动，最大工程收益 |
| ★★ | **#11 流式管线三层架构** | wenet, k2 | 接口契约清晰，v1→v2 不推翻重来 |
| ★★ | **#2 collate_fn stride pad** | wenet, slt | 预防 CNN 下采样对齐 bug，成本极低 |
| ★ | **#18 Gloss 后处理清洗** | slt | CTC 贪心碎片化，不改模型提 WER 2-3% |
| ★ | **#10 CTC Prefix Score** | espnet | beam=1 退化 = "更好的贪心"，不违 v1 约束 |

### 关键分歧

**#9 Strided Chunk + BiLSTM「半流式」方案**（k2 赞成 vs wenet 反对）

wenet 的判断——技术上可行但实际价值极低：
- 前向 LSTM 逐帧增量缓存 hidden state ✓
- 后向 LSTM 必须等 90 帧到齐 ✗（瓶颈）
- 前向先吐出的中间结果在后向完成后会被"修正" → 不可信
- **结论**：v1 老老实实 BiLSTM 全窗口推理，v2 一步到位切单向 LSTM + chunk cache，不要在中间态浪费精力。

**#10 CTC Prefix Score 是否违反 v1「no beam search」约束**（espnet vs wenet）

espnet 的反驳——**伪矛盾**：
- `beam_size=1` 时退化为"比贪心更聪明的贪心"（正确追踪 blank/非blank 两种状态）
- 计算量 O(T×vocab)，远小于全 attention beam search
- 这不是 beam search，是更好的 greedy。v1 不应因噎废食。
- wenet 同意技术上不冲突，但建议 v2 再做，先跑通 baseline。

### 组合效应 1+1>2

| 组合 | 效果 |
|------|------|
| #1 TemporalConv + #2 stride pad + #9 chunk cache | 完整流式特征提取管线（v2） |
| #9 chunk cache + #10 CTC prefix + #11 管线架构 + #12 多路状态 | wenet 生产级流式闭环（v2） |
| #15 NaN 存储 + #17 NaN% 分析 | 先量化问题 → 再系统防御 |
| #18 Gloss 清洗 + #10 CTC prefix + #20 Reduced BLEU | "评估引导解码"闭环，不依赖 LM |
| #22 CTC 转移字典 + #10 beam search + #13 attention rescoring | 解码能力递进升级，每层替换不推翻前一层 |

### 新洞察（Brainstorm 中涌现，原始条目未覆盖）

**A. NaN 语义分离模式**（landmark 提出）
```
存储层：np.full(shape, np.nan)   ← 保留"无检测"语义，用于审计
加载层：np.nan_to_num(nan=0.0)   ← Dataset.__getitem__ 中转换，保证模型数值稳定
```
两层各司其职。优于当前 `visibility<0.6 → 置零`（无法区分真实 (0,0) 和缺失）。

**B. 字符级 CTC 辅助任务**（slt 提出，基于 #21）
不需要额外标注——把词标签按字拆分（如 "高考/加油" → "高/考/加/油"），共享编码器加一个字符 CTC 头。提供更细粒度的时间对齐信号。成本 ~20 行。

**C. 1D Conv 下采样后的帧对齐 bug**（espnet 提出）
加入 #1 TemporalConv 后 `input_lengths` 计算错误是 CTC 训练最常见的坑——比 NaN 更难排查（loss 不报错，结果悄悄变差）。必须在 collate_fn 中同步更新 `input_lengths`。

**D. FSA 思想轻量版——显式 CTC 转移字典**（k2 提出）
```python
ctc_transition = {token_id: {"blank_self_loop": ..., "emit_arc": ...}}
```
约 20 行，为 v2 引入 N-gram LM 铺好结构（只需在转移字典上追加 compose）。

**E. CSL 中文虚词黑名单**（slt 提出，配合 #20 Reduced BLEU）
的/地/得/了/着/过/在/把/被/从/对/向/和/与/但/而/个/只/吧/吗/呢/啊/是/很/都/也/就/还（约 25 词）

**F. NaN% 分析的三层面 + 两级阈值**（landmark 提出）
- 样本级：每样本 hand0/hand1 NaN 帧占比
- 类级：每个词平均 NaN%（找出易遮挡词类）
- 帧位置级：序列头尾 vs 中间的 NaN 分布
- 阈值：警告线 20%，拒绝线 50%（hand1 放宽至 40%/70%）

**G. BiLSTM → 单向 LSTM 迁移路径**（wenet + k2 共识）
```
v1: BiLSTM 全窗口推理（训练+推理一致，最简单）
     ↓  不浪费时间在半流式方案上
v2: 单向 LSTM + chunk cache（真正的逐帧流式）
```
中间没有可行优化态。不要在前向增量+后向等全窗的混合方案上花精力。

### 解码能力升级路线（5 位共识）

```
v1: argmax → unique_consecutive → 去 blank（贪心解码）
     ↓  #10 CTC Prefix Score (beam=1→N，无需 LM，合理利用 CTC 拓扑)
v2: 显式 CTC 转移字典 + prefix beam search
     ↓  #22 引入 N-gram LM（在转移字典上追加 compose）
v2+: attention rescoring (#13) 对 nbest 做重排序
```
**核心原则**：每层替换不推翻前一层。解码器接口预留状态参数（#11），v1 填贪心实现，v2 换 beam search 不动其余层。

---

## 格式说明

每条启发记录格式：来源 → 核心思路 → 好处/代价 → 适用时机 → 状态。
- 🥩 **Raw** = 初次读取时的原始记录，未经讨论
- 🧠 **After Brainstorm** = 经 5 位 agent 交叉评审后有补充/修正
- 状态：「待讨论」「v1 采纳」「v2 保留」「不采纳」「参考」

## 来源索引

| 来源 | 条目 | 读取日期 | 阶段 |
|------|------|---------|------|
| TFNet-main | #1-6 | 2026-07-29 | Raw |
| MedSight | #7-8 | 2026-07-29 | Raw |
| reference/wenet | #9-13 | 2026-07-29 | Raw + 🧠 |
| reference/espnet | #14 | 2026-07-29 | Raw + 🧠 |
| reference/landmark-extraction | #15-17 | 2026-07-29 | Raw + 🧠 |
| reference/slt | #18-21 | 2026-07-29 | Raw + 🧠 |
| reference/k2 | #22-23 | 2026-07-29 | Raw + 🧠 |

## 工程教训（Lessons Learned）

### LL-1: worktree 文件对比时 Read 缓存的陷阱
- **日期**：2026-07-29
- **问题**：先在 worktree 里读了分支版 CLAUDE.md（精简版），切回根目录后 Read 工具返回 "file unchanged since your last Read"——但两个文件路径虽然同名内容不同，Read 走了缓存
- **后果**：误判两文件一致，遗漏 slt_how2sign 引用、SLR_Dataset 子目录细节、CHANGELOG 中间进展记录
- **教训**：跨 worktree 对比同名文件时，必须用  命令验证，不能靠 Read 工具输出做判断
- **修复**：3 条遗漏已补入分支；根目录文档用分支版本覆盖

