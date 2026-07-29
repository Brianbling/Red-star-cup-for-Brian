"""
主路模型训练脚本。
用法: python train.py
"""
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

BASE_DIR = Path("D:/red star project")
KEYPOINT_BASE = BASE_DIR / "CE-CSL/CE-CSL"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

sys.path.insert(0, str(BASE_DIR / "TFNet-main"))

from model import SLRModel
from dataset import KeypointDataset, collate_fn
from decode import ctc_greedy_decode, ctc_decode_text


def seed_torch(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_wer(pred_texts, label_texts):
    """简单 WER：基于字级别编辑距离（不依赖 TFNet WER.py 的复杂输出格式）。"""
    from WER import WerList
    hypotheses = [" ".join(t) for t in pred_texts]
    references = [" ".join(t) for t in label_texts]
    result = WerList(hypotheses=hypotheses, references=references)
    return result["wer"]


def main():
    seed_torch(0)

    with open(BASE_DIR / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    vocab_size = len(idx2word)
    blank = word2idx["<blank>"]

    print(f"词表大小: {vocab_size}, blank={blank}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # --- 数据集 ---
    train_set = KeypointDataset(KEYPOINT_BASE, "train", word2idx)
    dev_set = KeypointDataset(KEYPOINT_BASE, "dev", word2idx)

    print(f"训练集: {len(train_set)} 样本, 验证集: {len(dev_set)} 样本")

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=collate_fn,
    )

    # --- 模型 ---
    model = SLRModel(vocab_size=vocab_size).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # --- 损失 & 优化 ---
    ctc_loss_fn = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # --- 训练 ---
    epochs = 100
    best_wer = float("inf")
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            video = batch["video"].to(device)         # (T, 1, 84)
            input_lengths = batch["input_lengths"].to(device)
            label = batch["label"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            log_probs = model(video, input_lengths)   # (T, 1, vocab_size)

            loss = ctc_loss_fn(log_probs, label, input_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(loss.item())

        avg_loss = np.mean(losses) if losses else float("inf")
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        # --- 验证 ---
        model.eval()
        all_pred_texts = []
        all_label_texts = []
        dev_losses = []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validating"):
                video = batch["video"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                label = batch["label"].to(device)
                target_lengths = batch["target_lengths"].to(device)

                log_probs = model(video, input_lengths)

                loss = ctc_loss_fn(log_probs, label, input_lengths, target_lengths)
                if not (torch.isinf(loss) or torch.isnan(loss)):
                    dev_losses.append(loss.item())

                # 贪心解码
                pred_tokens = ctc_greedy_decode(log_probs, blank)
                pred_text = [idx2word[int(t)] for t in pred_tokens[0].tolist() if int(t) < len(idx2word)]

                label_tokens = label.tolist()
                label_text = [idx2word[t] for t in label_tokens]

                all_pred_texts.append(pred_text)
                all_label_texts.append(label_text)

        avg_dev_loss = np.mean(dev_losses) if dev_losses else float("inf")
        wer = compute_wer(all_pred_texts, all_label_texts)

        scheduler.step(wer)

        # --- 保存 ---
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "wer": wer,
            "vocab_size": vocab_size,
            "idx2word": idx2word,
            "word2idx": word2idx,
        }

        torch.save(checkpoint, CHECKPOINT_DIR / "last.pt")

        if wer < best_wer:
            best_wer = wer
            patience_counter = 0
            torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")
            print(f"  [NEW BEST] wer={wer:.2f}% → best.pt")
        else:
            patience_counter += 1

        print(f"  dev_loss={avg_dev_loss:.4f}, wer={wer:.2f}%, best_wer={best_wer:.2f}%, patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"训练完成。best_wer={best_wer:.2f}%")


if __name__ == "__main__":
    main()
