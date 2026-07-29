# CTC Blank 坍塌修复 — 完整实验路线

## 零、实验纪律

### 禁止事项
- ❌ 不碰 SR-CTC（已证明无效）
- ❌ 不碰 ad-hoc penalty（已证明有逻辑漏洞）
- ❌ 不回归无 bias / stride=2 的已知失败配置
- ❌ 不在一次实验中改变多个变量
- ❌ `zero_infinity=True` 在 P0 阶段绝对不开

### 每次实验必记录

| 指标 | 初始化后 | Epoch 1 | Epoch 3 | Epoch 5 |
|------|---------|---------|---------|---------|
| P(blank) % |         |         |         |         |
| unique classes/sample |  |         |         |         |
| zero_pred % |          |         |         |         |
| CTC loss (train) |     |         |         |         |
| CTC loss (dev) |       |         |         |         |
| WER (dev) |            |         |         |         |

---

## 一、P0 — 核心修复（优先级最高，最先执行）

### 目标
验证 "stride=4 + 正确的 blank bias 初始化 + 梯度裁剪" 能否让 BiLSTM+CTC 正常训练。

### 1.1 代码修改

#### src/model.py — bias 初始化

当前 Linear 层默认 bias=zeros，导致 P(blank) ≈ 1/478 ≈ 0.2%。需要显式设置：

```python
# 在 __init__ 中，Linear 层之后：
self.fc = nn.Linear(hidden_size, vocab_size)

# 正确方向：blank bias 比 others 高 +5.32
# P(blank) = σ(b_b - b_o) = σ(5.32) ≈ 0.3
nn.init.constant_(self.fc.bias, 0.0)         # 所有 bias 初始化为 0
self.fc.bias.data[blank_id] = 5.32            # blank_id = 0 (CTC 默认)

# 验证：训练前打印
# P(blank) = softmax(fc.bias)[0] 应 ≈ 0.3
```

**数学验证**：
P(blank) = exp(5.32) / (exp(5.32) + 477 * exp(0)) = 204.4 / (204.4 + 477) = 0.300 ✓

#### src/model.py — Stride=4 卷积

```python
# 当前 (stride=1)：
self.conv = nn.Conv1d(84, 256, kernel_size=3, padding=1)

# 修改为 (stride=4)：
self.conv = nn.Conv1d(84, 256, kernel_size=5, stride=4, padding=2)
# 输入 (B, 84, T) → 输出 (B, 256, T//4)
# kernel_size=5 保证 stride=4 时无间隙覆盖
```

#### src/train.py — 梯度裁剪 + 监控

```python
# 在 loss.backward() 和 optimizer.step() 之间：
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 每个 epoch 结束后打印诊断指标：
def compute_diagnostics(model, dataloader, blank_id=0):
    """计算 blank 占比、unique classes、zero_pred 比例"""
    model.eval()
    total_frames = 0
    blank_frames = 0
    total_unique = 0
    zero_pred_count = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            video = batch["video"].cuda()
            input_lengths = batch["input_lengths"]

            log_probs = model(video, input_lengths)

            for b in range(video.shape[1]):
                pred = log_probs[:input_lengths[b]//4, b, :].argmax(dim=-1)

                blank_frames += (pred == blank_id).sum().item()
                total_frames += (input_lengths[b]//4).item()

                unique = set(pred[pred != blank_id].cpu().numpy())
                total_unique += len(unique)

                decoded = ctc_greedy_decode(pred.unsqueeze(0), blank_id)
                if len(decoded) == 0:
                    zero_pred_count += 1
                total_samples += 1

    return {
        'p_blank': blank_frames / total_frames,
        'avg_unique': total_unique / total_samples,
        'zero_pred_pct': zero_pred_count / total_samples * 100
    }
```

#### src/train.py — zero_infinity=False（强制）

```python
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=False, reduction='mean')
```

### 1.2 训练配置

| 参数 | 值 | 理由 |
|------|-----|------|
| stride | 4 | T/L 68→13 |
| blank_bias | +5.32 | P(blank)≈30% |
| grad_clip | 5.0 | 防 inf loss 梯度爆炸 |
| zero_infinity | False | 必须暴露真实 loss |
| batch_size | 4 | stride=4 后内存省了 4x，可增大 |
| lr | 1e-3 | 不变 |
| optimizer | Adam | 不变 |
| scheduler | ReduceLROnPlateau | 不变 |
| max_epochs | 30 | 给足时间 |
| early_stop_patience | 15 | 放宽（可能需要更多 epoch） |

### 1.3 决策节点

#### Epoch 3 检查（快速验证方向）

| 观察 | 判断 | 行动 |
|------|------|------|
| P(blank) ∈ [20%, 50%] | ✅ 方向正确 | 继续训练到 Epoch 10+ |
| P(blank) > 80% | 坍塌仍在发生 | → 分支 A |
| P(blank) < 10% | bias 过头 | 降到 +3.0 重试 |
| avg_unique < 3 | 模型未分化 | → 分支 A |
| zero_pred > 50% | 解码全空 | → 分支 A |
| inf loss 频繁出现 | 梯度爆炸 | 降 lr 到 5e-4 |

#### Epoch 10 检查（判断收敛质量）

| 观察 | 判断 | 行动 |
|------|------|------|
| WER < 80% 且下降中 | ✅ 突破 | 继续到收敛 |
| WER 85-93% 但 P(blank) 健康 | 可能在学但慢 | 继续到 Epoch 20 |
| WER > 93% + P(blank) 偏离健康范围 | 又塌了 | → 分支 A |
| dev loss 不动 > 5 epoch | 停滞 | 降 lr 或 → 分支 A |

---

## 二、分支 A — Conv+ResNet 替代 BiLSTM（P0 fallback）

### 触发条件
P0 在 Epoch 3-5 时 P(blank) > 80% 且不下降。

### 设计

```
输入 (B, 84, T)
    │
    ▼
Conv1d(84→256, k=5, s=4, p=2)     # 降采样 T→T/4
    │
    ▼
BatchNorm1d + ReLU
    │
    ▼
ResBlock × 3:                      # 每个 block:
  Conv1d(256→256, k=3, p=1)        #   膨胀感受野
  BatchNorm + ReLU                 #   但不降采样
  Conv1d(256→256, k=3, p=1)
  BatchNorm
  └── + residual ──→ ReLU
    │
    ▼
Linear(256 → hidden=512)
    │
    ▼
BiLSTM(512→512, 2层) 或 直接 Linear(512 → vocab)
    │                           ↑ 先试这个
    ▼
Linear(hidden → vocab) + LogSoftmax
```

### 关键设计决策
- **ResBlock 替代纯 BiLSTM**：卷积的局部感受野天然限制时间平滑范围，比 BiLSTM 更难全局坍塌
- **保留 stride=4 卷积前置**：继续降采样
- **先试无 LSTM 版本**：如果纯 Conv+ResNet 能学到东西，说明 BiLSTM 确实是坍塌根源；如果仍然坍塌，说明问题比架构更深
- **参数量控制在 ~15M**：和当前模型 13.1M 可比

### 配置

| 参数 | 值 |
|------|-----|
| stride | 4（继承） |
| blank_bias | +5.32（继承） |
| grad_clip | 5.0（继承） |
| zero_infinity | False（继承） |
| 其他 | 同 P0 |

### 决策
- Epoch 3 P(blank) 健康 → 说明 BiLSTM 是罪魁祸首，继续此架构
- Epoch 3 仍然坍塌 → 分支 B

---

## 三、分支 B — Conformer Encoder

### 触发条件
Conv+ResNet 也无法阻止 blank 坍塌，或 P(blank) 健康但 WER 卡在 >85%。

### 设计

```
输入 (B, 84, T)
    │
    ▼
Conv2dSubsampling(84→256, stride=4)    # 降采样 + 升维
    │  T→T/4
    ▼
PositionalEncoding(256)
    │
    ▼
ConformerBlock × 4:                     # 每个 block:
  ├── FFN (256→1024→256)               #   d_model=256
  ├── MHSA (4 heads, 256 dim)          #   attention_heads=4
  ├── Conv1d(256→256, k=15)            #   macaron style
  └── LayerNorm × N
    │
    ▼
Linear(256 → vocab) + LogSoftmax
```

### 关键设计决策
- **不需要 BiLSTM**：Conformer 的 self-attention + convolution 已经覆盖了时序建模
- **d_model=256, 4 heads**：轻量配置，参数量 ~8-10M
- **Conv2dSubsampling 前置**：同时完成降采样和升维，比当前 1D Conv 更标准
- **参考实现**：Wenet 的 conformer/encoder.py 和 ESPnet 的 conformer/encoder.py

### 预期
Conformer 的 self-attention 天然产生非均匀对齐分布，blank 坍塌风险最低。这是架构层面的终极方案。

---

## 四、辅助策略（不单独实验，仅在主方案接近成功时叠加）

| 策略 | 触发条件 | 做法 | 优先级 |
|------|---------|------|--------|
| 数据筛选 | P(blank) 健康但 WER 下降慢 | 只训练 T/L < 20 的样本，长句子先学对齐 | 低 |
| warmup lr | 训练早期 loss 震荡 | 前 3 epoch lr 从 1e-5 线性升到 1e-3 | 低 |
| label smoothing | P(blank) 过高但 WER 有改善趋势 | CTC label smoothing ε=0.05 | 低 |
| weight decay | 过拟合迹象 | AdamW + wd=1e-4 | 低 |

---

## 五、完整决策树

```
P0: stride=4 + blank_bias=+5.32 + grad_clip=5.0
 │
 ├─[Epoch 3] P(blank) ∈ [20%, 50%] ─── 继续训练 ─── [Epoch 10-20] WER < 80%?
 │                                                         ├─ YES → 🎉 成功
 │                                                         └─ NO  → 调试（lr/warmup/数据）
 │
 ├─[Epoch 3] P(blank) > 80% ─── 分支 A: Conv+ResNet
 │                                    │
 │                                    ├─ P(blank) 健康 → 继续
 │                                    └─ P(blank) > 80% → 分支 B: Conformer
 │
 ├─[Epoch 3] P(blank) < 10% ─── bias 调低到 +3.0，重跑 P0
 │
 └─[Epoch 3] inf loss 频繁 ─── lr 降到 5e-4，重跑 P0
```

## 六、文件变更清单

| 文件 | 变更 |
|------|------|
| src/model.py | stride=4 Conv + Linear bias 初始化 |
| src/train.py | grad_clip + zero_infinity=False + 诊断日志 |
| src/model_resnet.py | 新增 — 分支 A 的 Conv+ResNet 模型 |
| src/model_conformer.py | 新增 — 分支 B 的 Conformer 模型 |
| checkpoints/ | 每个实验独立子目录：p0-bias5.32-s4/、branchA-resnet/、branchB-conformer/ |

## 七、成功标准

| 指标 | 最低通过线 | 良好 |
|------|-----------|------|
| WER (CE-CSL dev) | < 60% | < 30% |
| P(blank) 收敛后 | 20-60% | 20-40% |
| avg unique/sample | > 5 | > 10 |
| zero_pred % | < 5% | < 1% |

CE-CSL 是简单的 6000 句数据集。WER > 60% 说明模型基本没学到东西，不能转 CSL-Daily。
