# CTC Blank 坍塌 — 实验记录

文档说明：每次实验按下方模板记录，无论成功失败。

---

## 实验模板

```markdown
## 实验 X — [名称] — YYYY-MM-DD

### 配置快照
- 模型：SLRModel / ConvResNet / Conformer
- stride：4
- blank_bias：+5.32
- batch_size：4
- lr：1e-3
- zero_infinity：False
- grad_clip：5.0
- 特殊改动：（如有）

### 指标记录

| 指标 | 初始化后 | Epoch 1 | Epoch 3 | Epoch 5 | Epoch 10 |
|------|---------|---------|---------|---------|----------|
| P(blank) % |         |         |         |         |          |
| unique classes/sample |  |         |         |         |          |
| zero_pred % |          |         |         |         |          |
| CTC loss (train) |     |         |         |         |          |
| CTC loss (dev) |       |         |         |         |          |
| WER (dev) |            |         |         |         |          |

### 决策点

**Epoch 3**：
- 判断：✅ 继续 / ⚠️ 调整 / ❌ 失败
- 行动：

**Epoch 10**（如有）：
- 判断：
- 行动：

### 结论
（为什么成功/失败）
```

---

## 实验 1 — P0: stride=4 + blank_bias=+5.32 + BiLSTM

_待执行_

---

> 备注：每次实验的模型权重存放在 `checkpoints/{实验名}/best.pt`
