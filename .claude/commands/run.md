# 实时手语识别 — 运行技能

## 触发条件
用户提到：启动、运行、跑、demo、测试、训练、预处理 等相关操作。

## 前置检查

1. 确认 Python 环境可用：
   `python -c "import torch; import mediapipe; print('env ok')"`

2. 确认 CE-CSL 数据路径存在：
   `E:/CE-CSL/CE-CSL/video/train/`

3. 确认摄像头可用（推理模式）：
   `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

如果任一检查失败，报告具体缺失项并停止。

## 模式选择

根据用户意图选择以下模式之一：

### 模式 A：预处理（关键点提取）
用户说"预处理"、"提取关键点"、"extract"时触发。

**执行**：
1. 检查是否已有 `.npy` 关键点文件（`E:/CE-CSL/CE-CSL/keypoints/`），已存在则跳过
2. 对 train/dev/test 视频逐帧运行 MediaPipe Hands
3. 每帧输出 84 维向量（左右手各21点 × 2坐标 = 42点 × 2 = 84维），缺失手部或低置信度点置零
4. 每个视频保存为一个 `.npy`，文件名对应视频名
5. 打印进度条和 ETA

**输出**：`CE-CSL/CE-CSL/keypoints/{train,dev,test}/*.npy`

### 模式 B：训练
用户说"训练"、"train"、"fine-tune"时触发。

**执行**：
1. 确认预处理数据已就绪，否则先跑模式 A
2. 加载词汇表（从 CSV Gloss 列提取，按 `/` 分割去重）
3. 构建 Dataset/DataLoader（读 .npy + 标签），batch_size 默认 2（train.py 默认），按序列长度降序 padding
4. 初始化模型：1D Conv(84→256, stride=2×2 级联, 总降采样 /4) + BiLSTM(256→512, 2层,双向) + Linear(512→词表)
5. CTC Loss，Adam 优化器，初始 lr=0.001（比原 TFNet 更激进，因为模型小）
6. 训练 50-100 epochs，每个 epoch 在 dev 集上验证
7. 保存最佳 .pt 到 `D:/red star project/checkpoints/`

**关键参数**：
- hidden_size=512, num_layers=2, bidirectional=True
- lr=0.001, weight_decay=1e-4
- 早停 patience=15（train.py 默认，另有 ReduceLROnPlateau patience=5）

### 模式 C：推理（实时 demo）
用户说"启动"、"运行"、"demo"、"推理"、"测试"、"实时"时触发。

**执行**：
1. 确认模型权重存在（`D:/red star project/checkpoints/best.pt`），否则提示先训练
2. 加载模型、MediaPipe Hands、YOLOv8n（可选）
3. 初始化滑动窗口（deque, maxlen=90）、状态机（动态/静默/缓冲）
4. 打开摄像头，逐帧处理：
   - 预处理 → MediaPipe Hands → 置信度清洗 → 坐标归一化 → 追加窗口
   - 每5帧跑 YOLO 静态手势分类（可选）
   - 融合仲裁层判断走主路还是旁路
   - 主路：窗口序列送模型 → CTC 贪心解码 → 后处理去重 → 显示
   - 旁路：静态词 N 帧确认 → 显示
5. 叠加显示：摄像头画面 + 识别文本 + 关键点骨架
6. 按 Q 退出

**仲裁参数**（可运行时调整）：
- 速度阈值=0.01, 静默确认帧数=10, YOLO确认帧数=15, 手部丢失清缓存帧数=30

## 项目路径约定

| 用途 | 路径 |
|------|------|
| 数据集视频 | `E:/CE-CSL/CE-CSL/video/` |
| 关键点缓存 | `E:/CE-CSL/CE-CSL/keypoints/` |
| 模型权重 | `D:/red star project/checkpoints/` |

## 注意事项

- 训练时 batch_size 默认 2（CLAUDE.md 明确不要尝试大 batch，分组 padding 实验已证明退步）
- 预处理前先检查原视频路径，CE-CSL 下有双层嵌套目录
- 推理时如果无 GPU，模型推理用 CPU 也可（模型很小）
- YOLO 旁路如果卡顿，直接关掉，只保留主路
