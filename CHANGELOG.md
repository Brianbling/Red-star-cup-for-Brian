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

- **新增 reference/**：wenet、espnet、landmark-extraction、slt、slt_how2sign、k2 代码参考项目
- **INSPIRATION.md 大幅扩展**：从 8 条增至 23 条，覆盖 7 个来源。5 位 agent 完成交叉头脑风暴，生成 v1 优先级投票、共识/分歧报告、解码能力升级路线
- **CLAUDE.md 治理增强**：新增代码参考索引（模块→参考文件:行号映射）、Landmines 表、CSL-Daily 三步训练计划
- **新增数据集**：SLR_Dataset（CSL-2015，25K 孤立词 + 100 句连续）、CSL-Daily（E:/，~20K 句，下载中）
- **Phase 1 完成**：train 4972/4973（缺 train-01418），dev 515/515，test 500/500，合计 5987 .npy，~0.4 GB
- **多进程性能**：单进程 ~19h → 8 进程 ~3h。32 进程 16GB OOM → 降至 8 进程后稳定。预处理总帧数约 45 万帧
- **CLAUDE.md 精简整合**：314 行 → ~180 行，删除与 IMPLEMENTATION.md 重复的材料清单/工作量表/时间估算
