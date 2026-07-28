# 实时手语识别系统 — 实现计划

## 总览

```
Phase 0: 环境搭建         ██░░░░░░░░  0.5h   无需 GPU
Phase 1: 关键点预处理     ██████░░░░  4-6h   仅 CPU，可通宵挂机
Phase 2: 主路模型训练     ██████████  12-24h GPU，取决于 epoch 数
Phase 3: CTC 解码集成     ███░░░░░░░  1-2h   无需 GPU
Phase 4: 实时推理管线     ████████░░  4-6h   仅 CPU
Phase 5: 旁路 YOLO 分类   ████████░░  6-10h  GPU
Phase 6: 系统联调测试     ██████░░░░  3-4h   CPU + 摄像头
─────────────────────────────────────────
合计                                    约 30-50h（含训练等待）
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
- batch_size=1（序列长度差异大）

### 2.3 模型架构
```
Input (T, 84)
  → 1D Conv (kernel=3, 84→256) + ReLU + BN
  → BiLSTM (input=256, hidden=512, num_layers=2)
  → Linear (512 → vocab_size+1)   # +1 for blank
  → LogSoftmax
```

### 2.4 训练
- Loss: CTC Loss，blank=0
- Optimizer: Adam, lr=0.001
- Scheduler: ReduceLROnPlateau, patience=5
- Early stopping: patience=10 on dev WER
- Epochs: 50-100
- 每个 epoch 后跑 dev set 验证 WER
- 保存 best.pt + last.pt 到 `checkpoints/`

### 2.5 时间估算
| 因素 | 估算 |
|------|------|
| 模型参数 | **13.1M**（初始预估 ~5M 偏低） |
| 每 epoch 训练 | ~10-15min（batch_size=1，6000 样本） |
| 50 epoch | ~10-12h |
| 早停可能提前 | ~30 epoch / ~6h |
| 建议 | 先跑 3-5 epoch 看 loss 下降趋势，正常再全量 |

### 2.6 风险
- **CTC 不收敛**：最常见的坑。先确保 blank 位置正确，input_lengths 计算正确（BiLSTM 不降时间维，input_lengths = T）
- **全零帧过多**：手部丢失严重的话，考虑加 mask 机制
- **WER 很高（>80%）**：v1 预期就是高，不用慌。可以加少量 TFNet 的 Transformer 层试一下

---

## Phase 3 — CTC 解码集成（~1-2h，CPU）

### 3.1 贪心解码（自己写）
```python
def ctc_greedy_decode(logits):
    # logits: (T, vocab_size+1)
    # 1. argmax 每帧
    # 2. 合并连续相同 token
    # 3. 去掉 blank(0)
    # 返回 token id 序列
```
约 30 行代码。如果 ctc_decoders 能编译成就用它的 C++ 版本（更快）。

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
- **模型推理延迟**：BiLSTM 序列长度 90 时推理应该 <10ms，问题不大
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
- BiLSTM 双向延迟 ~150ms — 目测是否可接受
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
