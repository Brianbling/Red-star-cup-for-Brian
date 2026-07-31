# 实时手语识别系统 — 实现计划

> 状态：Phase 0-3 已完成（2026-07-31）。Phase 2 结论：WER=66.58% 受 ~5K 数据量限制。
> 后续路线见 `docs/temporary/实验方案.md`（孤立词预训练 → 数据扩展 → 视觉评估）。

## 总览

```
Phase 0: 环境搭建         ██████████  0.5h   ✅ 完成
Phase 1: 关键点预处理     ██████████  4-6h   ✅ 完成（5987 .npy）
Phase 2: 主路模型训练     ██████████  12-24h ✅ 完成（WER 66.58%，数据瓶颈）
Phase 3: CTC 解码集成     ██████████  1-2h   ✅ 完成（贪心解码）
Phase 4: 实时推理管线     ████████░░  4-6h   待开始
Phase 5: 旁路 YOLO 分类   ████████░░  6-10h  待开始
Phase 6: 系统联调测试     ██████░░░░  3-4h   待开始
```

---

## Phase 0 — 环境搭建（~0.5h，CPU）

### 0.1 创建目录结构
```bash
mkdir -p checkpoints
mkdir -p CE-CSL/CE-CSL/keypoints/{train,dev,test}
```

### 0.2 写 `requirements.txt`
```
torch>=2.0
mediapipe==0.10.35
opencv-python
numpy
ultralytics
```
说明：ctc_decoders 可能需要编译 SWIG，先留坑，不行就手写贪心解码。

### 0.3 验证依赖
```bash
python -c "import torch, mediapipe, cv2, numpy; print('ok')"
```
不通过就补装，不通过不进下一阶段。

---

## Phase 1 — 关键点预处理（~4-6h，仅 CPU）

### 目标
把 CE-CSL 全部 6000 个视频逐帧跑 MediaPipe Hands，保存为 `.npy` 关键点文件。

### 产出文件
```
CE-CSL/CE-CSL/keypoints/train/<video_id>.npy   # shape: (T, 84) 即 42点×2坐标
CE-CSL/CE-CSL/keypoints/dev/<video_id>.npy     # visibility<0.6 的关键点已置零
CE-CSL/CE-CSL/keypoints/test/<video_id>.npy    # 手部丢失帧全零向量
```

### 关键实现细节
- 每帧提取左右手各 21 点 (x,y)，即 84 维向量
- visibility < 0.6 的坐标置零（置信度清洗）
- 手部完全丢失时填全零（不丢帧，保持帧序号对齐）
- 已处理的视频跳过（支持断点续跑）
- 每个视频跑完立即存盘，不内存积压
- 打印进度：`[1234/6000] train/video_xxx.mp4 (68 frames, 2.7s)`

### 时间估算
- MediaPipe Hands 在 CPU 上约 50-100fps（取决于 CPU）
- 6000 个视频 × 平均 75 帧 = 450,000 帧
- 450,000 / 80fps ≈ 5400s ≈ **1.5h**（好的 CPU）到 **5h**（普通 CPU）
- 建议：晚上挂机跑，第二天检查

### 注意事项
- CE-CSL 目录嵌套了两层（CE-CSL/CE-CSL/），路径别写错
- train-01418 缺视频，CSV 行需跳过
- 先用 10 个视频验证流程正确性再全量跑

---

## Phase 2 — 主路模型训练（~12-24h，需 GPU）

### 2.1 词汇表构建
- 从 CE-CSL CSV 的 Gloss 列提取所有词
- 按 `/` 分割，去重，去空，去纯标点
- 加 `<blank>` 作为 CTC 的 blank token
- 生成 `vocab.json`：`{"<blank>": 0, "2023": 1, "高考": 2, ...}`
- 实际词表大小：**3515 tokens**（初始预估 500-1000 偏低）

### 2.2 Dataset / DataLoader
- 读取 `.npy` 关键点文件 → tensor (T, 84)
- 读取对应 CSV 标签 → token ids → tensor (L,)
- `collate_fn`：按 T 降序排列，pad 到 batch_max_len
- batch_size=2（train.py 默认）。分组 padding (bs=4) 实验证明退步 2.3pp，序列长度差异本身是有效正则化

### 2.3 模型架构
```
Input (T, 84) 或 (T, 660)  # 660 为 kp+visual concat 模式
  → 1D Conv (kernel=3, stride=4, 84→256) + ReLU  # T→T/4
  → BiLSTM × 2 (hidden_size=512, bidirectional)
  → Linear (1024 → vocab_size)
  → LogSoftmax
```

实际参数量：13.5M（全量 3515 词表）/ 10.4M（top-479 词表）。使用 BiLSTM 而非 Conformer（尽管文档曾计划升级为 Conformer，核心代码未实现）。

FC 层 blank token bias 初始化为 +5.32，确保 P(blank) ~ 0.995——这是 CTC 训练稳定性的必需设计，非 hack。不给正 bias 时 P(blank) 会在 1 epoch 内从 0 → 1.0，导致大幅 blank 坍塌。

**视觉融合模式**（通过 `--visual-fusion` 控制）：
- `none`：纯 kp-only，84d 输入
- `raw`：直接 concat kp(84d) + visual(576d) = 660d 输入
- `projected`：双线性投影 kp→128d + visual→128d → concat 256d 输入

**实验结论（2026-07-30）**：raw concat collapse=0% 但 WER=84.79%（比 kp-only 66.58% 差），projected 更差（91.61%）。MobileNetV3-Small ImageNet 特征编码物体类别，对 CSL 手语无效。视觉融合路线暂搁置。

### 2.4 训练
- Loss: CTC Loss，blank=0，`zero_infinity=False`（不要用 True，会掩盖 blank 坍塌）
- Optimizer: Adam, lr=0.001, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau, patience=5, factor=0.5
- Early stopping: patience=15 on dev WER（当前实现）
- Epochs: 50-100
- blank penalty（threshold=0.65, weight=20.0）+ 熵正则（weight=0.01）
- 每个 epoch 后跑 dev set 验证 WER + S/D/I + collapse%
- 保存 best.pt + last.pt 到 `checkpoints/`

### 2.4.1 实验结果（2026-07-28 → 2026-07-31）
| 日期 | 改动 | Best WER |
|------|------|----------|
| 2026-07-28 | 原始 baseline（BiLSTM, 原始坐标, batch=1） | 93.15% |
| 2026-07-29 | + 坐标归一化 / top-478 子词表 / blank_bias | ~93-95%（blank 坍塌） |
| 2026-07-30 | + stride=4 + blank penalty + 熵正则 + activity detection | 66.85% |
| 2026-07-30 | + clean_word 一致性修复重训 | 66.58% |
| 2026-07-31 | 零成本优化（分组 padding / 时序增强 / beam search） | 全部退步或不改善 |

**最终结论**：~5K 数据量是 WER 瓶颈（66.58%），非模型架构或解码策略。突破需更多数据（SLR_Dataset 孤立词预训练 / CSL-Daily 数据扩展，见 docs/temporary/实验方案.md）。

### 2.5 时间估算
| 因素 | 估算 |
|------|------|
| 模型参数 | 13.5M（3515 词表）/ 10.4M（top-479 词表） |
| 每 epoch 训练 | ~10-15min（batch_size=2，~5K 样本） |
| 50 epoch | ~10-12h |
| 早停可能提前 | ~30 epoch / ~6h |
| 建议 | 先跑 3-5 epoch 看 loss 下降趋势，正常再全量 |

### 2.6 风险
- **CTC blank 坍塌**：核心风险（T/L=68:1）。已通过 stride=4 + blank_bias=+5.32 + blank penalty + 熵正则 + activity detection 解决
- **全零帧过多**：手部丢失严重的话，activity_detect.py 已实现切分
- **WER 平台（~66%）**：数据量瓶颈，非模型问题。工程优化（padding/增强/beam search）均已证明无效

---

## Phase 3 — CTC 解码集成（~1-2h，CPU）

### 3.1 贪心解码（自己写，已完成 src/decode.py）
```python
def ctc_greedy_decode(logits):
    # logits: (T, vocab_size)
    # 1. argmax 每帧
    # 2. unique_consecutive 合并连续相同 token
    # 3. 去掉 blank(0)
    # 返回 token id 序列
```
约 30 行代码。已实现。beam search 实验证明与贪心等价（P(blank)≈0.77），无需替换。

### 3.2 后处理
- 连续重复字符合并
- blank 过滤
- 空标签过滤（无输出的帧）
- 可选的平滑：连续 N 帧无变化才更新输出文本，减少闪烁

### 3.3 验证
- 拿一条训练集视频，贪心解码 vs 标签，确认输出格式正确
- 不需要跑 WER 评估（WER 评估用现有的 `TFNet-main/WER.py`）

---

## Phase 4 — 实时推理管线（~4-6h，CPU）

### 4.1 模块拆分（4 个待写文件 + 2 个已完成）

`model.py` 和 `decode.py` 已在 Phase 2/3 中完成（`src/model.py`、`src/decode.py`），可直接导入使用。剩余 4 个文件：

#### `preprocess.py` — 帧预处理 + 特征提取
```
输入: 摄像头 BGR 帧
输出: 84 维关键点向量（或 None，手部丢失时）
逻辑:
  1. resize 256×256, BGR→RGB
  2. MediaPipe Hands 推理
  3. 提取左右手 42 点 (x,y)
  4. visibility < 0.6 → 置零
  5. 以手腕间距做坐标归一化
```

#### `window.py` — 滑动窗口
```
输入: 每帧 84 维向量
输出: (T_win, 84) 窗口内序列
逻辑:
  deque maxlen=90 (3s@30fps)
  append 每帧，自动出队旧帧
  get_sequence() 返回完整窗口
```

#### `arbitration.py` — 融合仲裁状态机
```
输入: 当前帧关键点, YOLO 分类结果, MediaPipe 置信度
输出: "动态"/"静默"/"缓冲" 状态 + 选择主路或旁路
逻辑: 按 ARCHITECTURE.md 的三态状态机实现
参数: 速度阈值=0.01, 静默T=10帧, YOLO确认N=15帧
```

#### `model.py` — 时序模型
```
与训练时同一架构（Phase 2.3）
加载 best.pt
推理模式 forward(x): x(T,84) → logits(T, vocab+1)
```

#### `decode.py` — CTC 解码 + 后处理
```
复用 Phase 3 的贪心解码 + 后处理逻辑
```

#### `inference.py` — 主循环
```
摄像头 → preprocess → window → model → decode → 叠加显示 → 按Q退出
                                  ↑
                            arbitration
                                  ↑
                             YOLO(每5帧)
```

### 4.2 时间估算
- 6 个模块，每个 0.5-1h
- 联调 1-2h
- 总计 4-6h

### 4.3 风险
- **MediaPipe 实时推理延迟**：实际测试下来如果 >50ms/帧，30fps 跟不住，需要降分辨率或跳帧
- **模型推理延迟**：BiLSTM 序列长度 22（90 帧窗口 / stride 4）时推理应该 <10ms，问题不大
- **YOLO 每 5 帧**：如果 YOLOv8n 推理 >50ms，可以改为每 10 帧

---

## Phase 5 — 旁路 YOLO 静态手势分类（~6-10h，需 GPU）

### 5.1 数据准备
- CSL_basic_dataset（235 词）+ CSL_common_dataset（863 词）
- 合并去重，按 8:2 随机划分 train/val
- 从每个视频等间隔抽 5 帧（或更多，视视频长度）
- 生成 YOLO 格式：`images/train/<word>/frame_001.jpg` + 对应 txt 标签
- 类别数 ≈ 1098（两个数据集大类数）

### 5.2 训练 YOLOv8n-cls
- 用 ultralytics 的 classification 模式（不是 detection，指手势分类）
- Pretrained: yolov8n-cls.pt（ImageNet 预训练）
- epochs: 50, imgsz: 224, batch: 32
- 导出 ONNX 用于推理

### 5.3 集成到推理管线
- inference.py 中每 5 帧跑一次 YOLO 分类
- 分类结果送入 arbitration.py 的旁路逻辑
- 如果关闭 YOLO（低算力模式），旁路静默

### 5.4 时间估算
| 步骤 | 时间 |
|------|------|
| 数据准备、抽帧 | 1-2h |
| 训练 | 4-6h |
| 集成推理 | 1-2h |

### 5.5 风险
- **1098 类太多**：部分词只有 1 个样本，分类精度会很低。可以设置置信度阈值，<0.5 的不触发
- **与主路词表重复**：YOLO 输出的词可能在主路 CTC 词表中也有，仲裁层用"主路优先"策略解决

---

## Phase 6 — 系统联调测试（~3-4h，CPU + 摄像头）

### 6.1 功能测试
- [ ] 摄像头打开正常
- [ ] MediaPipe 关键点实时画出
- [ ] 滑动窗口帧数显示
- [ ] 仲裁状态实时显示（动态/静默/缓冲）
- [ ] 主路输出文本叠加到画面
- [ ] 旁路静态词输出（如果开启 YOLO）
- [ ] 按 Q 退出

### 6.2 延迟测量
- 逐模块计时，找出瓶颈
- 目标：端到端 <200ms
- 如果超标：降分辨率、跳帧、关 YOLO

### 6.3 场景测试
- 正常手语速度
- 极慢动作（测试静默切换）
- 停顿后继续（测试缓冲态）
- 手部进出画面（测试缓存清空）
- 无手部时（测试无输出不崩溃）

### 6.4 已知限制确认
- BiLSTM 双向延迟 ~150-300ms — 目测是否可接受
- 无语言模型 — 观察输出抖动程度
- 近形手势 — 记录混淆样例，v2 针对性优化

---

## 附录：建议执行顺序

```
开始 → Phase 0 (环境) → Phase 1 (预处理，挂机) → Phase 2 (训练，挂机)
                                                       ↓
                                              同时做 Phase 3 (解码)
                                                       ↓
                                              Phase 4 (推理管线)
                                                       ↓
                                              Phase 5 (YOLO旁路)
                                                       ↓
                                              Phase 6 (联调测试) → v1 完成
```

**最关键的三步**：Phase 1（预处理正确，后续全依赖它）、Phase 4（推理管线，工程复杂度最高）、Phase 6（联调，暴露所有集成问题）。

**最耗时但最简单**：Phase 1 和 Phase 2，挂机就行。
