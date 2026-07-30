"""
诊断脚本 — CTC Blank 坍塌根因分析
诊断 A: 区分"对齐正确的 blank"和"全面坍塌的 blank"
诊断 B: 非 blank 预测的 token 多样性
诊断 C: 关键点帧间位移分布
诊断 D: 标签抽查
用法: python diagnose.py
"""
import json
import sys
import numpy as np
import torch
from pathlib import Path
from collections import Counter

BASE_DIR = Path("D:/red star project")
SCRIPT_DIR = Path("D:/red star project/.claude/worktrees/p0-ctc-blank-repair")

sys.path.insert(0, str(SCRIPT_DIR / "src"))
sys.path.insert(1, str(BASE_DIR / "TFNet-main"))

from model_conformer import Conformer
from dataset import KeypointDataset, collate_fn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(BASE_DIR / "vocab_top478.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
word2idx = vocab["word2idx"]
idx2word = vocab["idx2word"]
blank_id = word2idx["<blank>"]

print("=" * 60)
print("诊断 A: 逐帧 argmax 分布（对齐 vs 坍塌）")
print("=" * 60)

checkpoint = torch.load(BASE_DIR / "checkpoints" / "best.pt", map_location=device, weights_only=False)
model = Conformer(vocab_size=len(idx2word), blank_id=blank_id).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dev_set = KeypointDataset(BASE_DIR / "CE-CSL/CE-CSL", "dev", word2idx)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

all_non_blank_positions = []
all_non_blank_tokens = []
sample_reports = []

with torch.no_grad():
    for i, batch in enumerate(dev_loader):
        if i >= 10:
            break
        video = batch["video"].to(device)
        input_lengths = batch["input_lengths"]
        label = batch["label"]
        target_lengths = batch["target_lengths"]
        video_id = batch["video_ids"][0]

        log_probs = model(video, input_lengths)
        T_rnn = (input_lengths[0] // 4).item()
        pred = log_probs[:T_rnn, 0, :].argmax(dim=-1).cpu().numpy()

        label_tokens = label.tolist()
        label_text = [idx2word[t] for t in label_tokens]

        non_blank_idx = np.where(pred != blank_id)[0]
        non_blank_tokens_list = pred[non_blank_idx].tolist()

        if len(non_blank_idx) > 0:
            gaps = np.diff(non_blank_idx)
            clustered = (gaps <= 3).sum() / max(len(gaps), 1)
            all_non_blank_positions.extend((non_blank_idx / T_rnn).tolist())
            all_non_blank_tokens.extend(non_blank_tokens_list)

            sample_reports.append({
                "video_id": video_id,
                "T_raw": input_lengths[0].item(),
                "T_rnn": T_rnn,
                "n_non_blank": len(non_blank_idx),
                "n_spikes": len(np.where(gaps > 3)[0]) + 1 if len(gaps) > 0 else 1,
                "clustered_ratio": f"{clustered:.0%}",
                "label": "/".join(label_text),
            })
        else:
            sample_reports.append({
                "video_id": video_id,
                "T_raw": input_lengths[0].item(),
                "T_rnn": T_rnn,
                "n_non_blank": 0,
                "n_spikes": 0,
                "label": "/".join(label_text),
            })

print(f"\n{'video_id':>18s}  T_raw  T_rnn  non_blank  spikes  clustered  label")
print("-" * 80)
for r in sample_reports:
    spikes_str = str(r['n_spikes']) if r['n_spikes'] > 0 else "-"
    clustered_str = r.get('clustered_ratio', '-')
    print(f"{r['video_id']:>18s}  {r['T_raw']:5d}  {r['T_rnn']:4d}  {r['n_non_blank']:9d}  {spikes_str:>6s}  {clustered_str:>10s}  {r['label']}")

total_non_blank = len(all_non_blank_positions)
print(f"\n总计: {total_non_blank} 个非 blank 帧 (来自 10 个样本)")
if total_non_blank > 0:
    print(f"非 blank 位置中位数: {np.median(all_non_blank_positions):.2f} (0=开头, 1=结尾)")

print("\n" + "=" * 60)
print("诊断 B: 非 blank token 多样性（全 dev 集）")
print("=" * 60)

all_tokens = []
total_samples = 0
zero_pred_count = 0
token_counter = Counter()

with torch.no_grad():
    for batch in dev_loader:
        video = batch["video"].to(device)
        input_lengths = batch["input_lengths"]

        log_probs = model(video, input_lengths)
        for b in range(video.shape[1]):
            T_b = (input_lengths[b] // 4).item()
            pred = log_probs[:T_b, b, :].argmax(dim=-1).cpu().numpy()

            tokens = pred[pred != blank_id]
            if len(tokens) == 0:
                zero_pred_count += 1
            else:
                all_tokens.extend(tokens.tolist())
                token_counter.update(tokens.tolist())
            total_samples += 1

unique_tokens = len(token_counter)
print(f"total_samples: {total_samples}")
print(f"zero_pred: {zero_pred_count} ({zero_pred_count/total_samples:.1%})")
print(f"unique non-blank tokens: {unique_tokens}")
print(f"total non-blank frames: {len(all_tokens)}")
if token_counter:
    print(f"Top-10 tokens by frequency:")
    for tok, cnt in token_counter.most_common(10):
        print(f"  {idx2word[tok]:>10s} (id={tok:3d}): {cnt:5d} ({cnt/len(all_tokens):.1%})")

print("\n" + "=" * 60)
print("诊断 C: 关键点帧间位移分布（训练集抽样）")
print("=" * 60)

train_set = KeypointDataset(BASE_DIR / "CE-CSL/CE-CSL", "train", word2idx)
np.random.seed(0)
sample_indices = np.random.choice(len(train_set), size=min(500, len(train_set)), replace=False)

frame_diffs = []
for idx in sample_indices:
    item = train_set[idx]
    kp = item["keypoints"].numpy()  # (T, 84)
    if kp.shape[0] <= 1:
        continue
    diff = np.linalg.norm(kp[1:] - kp[:-1], axis=1)
    frame_diffs.extend(diff.tolist())

frame_diffs = np.array(frame_diffs)
print(f"抽样样本数: {len(sample_indices)}")
print(f"总帧间位移对: {len(frame_diffs):,}")
print(f"中位数位移: {np.median(frame_diffs):.6f}")
print(f"均值位移: {np.mean(frame_diffs):.6f}")
print(f"位移 < 0.001 的比例: {(frame_diffs < 0.001).mean():.1%}")
print(f"位移 < 0.005 的比例: {(frame_diffs < 0.005).mean():.1%}")
print(f"位移 < 0.01 的比例: {(frame_diffs < 0.01).mean():.1%}")
for pct in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {pct}分位数: {np.percentile(frame_diffs, pct):.6f}")

print("\n" + "=" * 60)
print("诊断 D: 标签抽查 (5 个训练集样本)")
print("=" * 60)

for idx in np.random.choice(len(train_set), size=5, replace=False):
    item = train_set[idx]
    kp = item["keypoints"]
    label = item["label"]
    label_text = [idx2word[t.item()] for t in label]
    print(f"  {item['video_id']}: T={kp.shape[0]}, L={len(label_text)}, label={'/'.join(label_text)}")
