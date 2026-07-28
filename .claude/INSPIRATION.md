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

## 格式说明
每条启发记录：来源 → 核心思路 → 好处/代价 → 适用时机 → 状态。状态分为"待讨论"/"v1 采纳"/"v2 保留"/"不采纳"。
