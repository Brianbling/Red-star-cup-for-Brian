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
- **Phase 1 进度**：train 1796/4972 (36.1%)，最新输出 23:53，正常运行中
- 清理无关进程：Anaconda Navigator (pythonw.exe x2)
- **文档校正**：ARCHITECTURE.md 训练流程描述（独立实现非复用 TFNet）、IMPLEMENTATION.md 词表 3515（非~500-1000）、模型 13.1M（非~5M）、Phase 4 模块拆分更新

## 2026-07-29

### Phase 1 — 关键点预处理 [完成]
- **全量完成**：train 4972/4973（缺 train-01418），dev 515/515，test 500/500。合计 5987 .npy，~0.4 GB
- **多进程性能**：单进程 ~19h → 8 进程 ~3h。32 进程 16GB OOM → 降至 8 进程后稳定。预处理总帧数约 45 万帧
- 输出格式：(T, 84) float32，存原始 MediaPipe 图像归一化坐标（未做手腕归一化）

### Phase 2 — 首次训练 [失败，需重训]
- Epoch 1-15，early stopping(patience=10) 触发。Best WER **93.15%**（Epoch 5）
- train loss 7.24→5.99，dev loss ~6.3-6.5 不动
- **根因**：坐标未做手腕归一化。虽然 MediaPipe 已做图像归一化 [0,1]，但不同视频中手部位置/距离不同（手部跨度差异 4.94x，平移变化显著），同一手势在不同视频中产生完全不同的 84 维向量。模型学到的是"绝对坐标→gloss"的视频指纹映射，而非"手部构型→gloss"的语义映射
- **次要问题**：batch_size=1 下 BatchNorm1d 退化为噪声、CTC T/L 比例 33:1 过大、词表 3515 tokens 中 49.3% 只出现 1 次

### 四专家联合诊断（2026-07-29）
- 启动 4 位专家 agent（训练数据/模型架构/预处理归一化/训练策略）并行分析 + 交叉讨论
- 核心共识：坐标归一化缺失是唯一主因，修复后预期 WER 降至 25-40%
- 发现关键 bug：`build_vocab.py` 用 `clean_word()` 去掉了词尾数字后缀（"高考1"→"高考"），但 `dataset.py._gloss_to_ids()` 未调用 `clean_word()`，导致 **6.0% 的 token（1701/28272）被静默丢弃**
- 纠正数据认知：真正的 84 维全零帧率 5.0%（非此前估计的 29%），但训练集前后分布极不均衡（前 100 文件 0.8% vs 后 100 文件 39.7%）
- 归一化方案选定：手部跨度（手腕→中指 MCP 距离）做分母，每视频固定 scale（中位数），先改 `dataset.py` 在线归一化验证（0 重预处理），确认有效后再改 `preprocess_keypoints.py` 做离线归一化

### 修复计划（按优先级）
- **P0**：`dataset.py` 加 `clean_word()`（修复 6% token 丢弃）+ BatchNorm1d→LayerNorm（B=1 下 BN 是 bug）+ gradient clip(max_norm=5.0)
- **P1**：坐标归一化（dataset.py 手部跨度在线归一化）+ top-381 词表快速验证管线
- **P2**：全词表重训（batch_size=8, lr=4e-4, warmup 5 epoch, early stop 改用 dev loss, patience=25）
- **P3**：`preprocess_keypoints.py` NaN 改造 + 全量重跑，Conv stride=2 消融实验

### 文档维护
- **CLAUDE.md 治理增强**：新增代码参考索引（模块→参考文件:行号映射）、Landmines 表、CSL-Daily 三步训练计划
- **INSPIRATION.md 大幅扩展**：从 8 条增至 23 条，覆盖 7 个来源。5 位 agent 完成交叉头脑风暴
- **ARCHITECTURE.md 修正**：过时的 TFNet 复用清单（仅 WER.py）、时序模型描述（84 维非 42 维、未复用 BiLSTM.py）
- **新增数据集**：SLR_Dataset（CSL-2015，25K 孤立词 + 100 句连续）、CSL-Daily（E:/，~20K 句，下载中）
