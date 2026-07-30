"""
主路模型训练脚本。
用法: python train.py                    # 全量词表(3515)
       python train.py --vocab vocab_top478.json  # 子词表快速验证
"""
import json
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

BASE_DIR = Path("D:/red star project")
SCRIPT_DIR = Path("D:/red star project/.claude/worktrees/p0-ctc-blank-repair")
KEYPOINT_BASE = BASE_DIR / "CE-CSL/CE-CSL"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

sys.path.insert(0, str(SCRIPT_DIR / "src"))
sys.path.insert(1, str(BASE_DIR / "TFNet-main"))

from model_conformer import Conformer
from dataset import KeypointDataset, collate_fn
from decode import ctc_greedy_decode


def seed_torch(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_wer(pred_texts, label_texts):
    from WER import WerList
    hypotheses = [" ".join(t) for t in pred_texts]
    references = [" ".join(t) for t in label_texts]
    result = WerList(hypotheses=hypotheses, references=references)
    return result["wer"]


def compute_diagnostics(model, dataloader, blank_id, idx2word, ctc_input_divisor, device):
    model.eval()
    total_frames = 0
    blank_frames = 0
    total_unique = 0
    zero_pred_count = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            video = batch["video"].to(device)
            input_lengths = batch["input_lengths"]
            label = batch["label"]
            target_lengths = batch["target_lengths"]

            log_probs = model(video, input_lengths)
            ctc_input_lengths = (input_lengths // ctc_input_divisor).clamp(min=1)

            for b in range(video.shape[1]):
                T_b = ctc_input_lengths[b].item()
                pred = log_probs[:T_b, b, :].argmax(dim=-1)

                blank_frames += (pred == blank_id).sum().item()
                total_frames += T_b

                unique = set(pred[pred != blank_id].cpu().numpy())
                total_unique += len(unique)

                decoded = ctc_greedy_decode(pred.unsqueeze(0), blank_id)
                if len(decoded[0]) == 0:
                    zero_pred_count += 1
                total_samples += 1

    return {
        'p_blank': blank_frames / max(total_frames, 1),
        'avg_unique': total_unique / max(total_samples, 1),
        'zero_pred_pct': zero_pred_count / max(total_samples, 1) * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, default="vocab.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    args = parser.parse_args()

    seed_torch(0)

    with open(BASE_DIR / args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    vocab_size = len(idx2word)
    blank = word2idx["<blank>"]

    print(f"词表: {args.vocab}, 大小: {vocab_size}, blank={blank}")
    print(f"batch_size: {args.batch_size}, AMP: {args.amp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    train_set = KeypointDataset(KEYPOINT_BASE, "train", word2idx)
    dev_set = KeypointDataset(KEYPOINT_BASE, "dev", word2idx)
    print(f"训练集: {len(train_set)} 样本, 验证集: {len(dev_set)} 样本")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=collate_fn,
    )

    model = Conformer(vocab_size=vocab_size, blank_id=blank).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    bias_values = model.fc.bias.data[:10].tolist()
    print(f"FC bias 前10: {[f'{v:.2f}' for v in bias_values]}")
    with torch.no_grad():
        init_probs = torch.softmax(model.fc.bias, dim=-1)
        print(f"P(blank|bias)={init_probs[blank].item():.3f}, "
              f"P(top-5 non-blank)={init_probs[1:6].tolist()}")

    ctc_loss_fn = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    ctc_input_divisor = 4

    epochs = 30
    best_wer = float("inf")
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            video = batch["video"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            label = batch["label"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            with torch.autocast("cuda", enabled=args.amp):
                log_probs = model(video, input_lengths)
                ctc_input_lengths = (input_lengths // ctc_input_divisor).clamp(min=1)
                loss = ctc_loss_fn(log_probs, label, ctc_input_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            losses.append(loss.item())

        if not losses:
            print(f"Epoch {epoch+1}: 所有 loss 为 inf/nan，跳过")
            continue

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        diag = compute_diagnostics(model, dev_loader, blank, idx2word, ctc_input_divisor, device)
        print(f"  [Diag] P(blank)={diag['p_blank']:.1%}, "
              f"uniq/sample={diag['avg_unique']:.1f}, "
              f"zero_pred={diag['zero_pred_pct']:.1f}%")

        model.eval()
        all_pred_texts = []
        all_label_texts = []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validating"):
                video = batch["video"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                label = batch["label"]
                target_lengths = batch["target_lengths"]

                with torch.autocast("cuda", enabled=args.amp):
                    log_probs = model(video, input_lengths)

                pred_tokens = ctc_greedy_decode(log_probs, blank)
                for b in range(video.shape[1]):
                    pred_text = [idx2word[int(t)] for t in pred_tokens[b].tolist() if int(t) < len(idx2word)]
                    label_start = sum(target_lengths[:b])
                    label_end = label_start + target_lengths[b]
                    label_tokens = label[label_start:label_end].tolist()
                    label_text = [idx2word[t] for t in label_tokens]
                    all_pred_texts.append(pred_text)
                    all_label_texts.append(label_text)

        wer = compute_wer(all_pred_texts, all_label_texts)
        scheduler.step(wer)

        total_pred_tokens = sum(len(p) for p in all_pred_texts)
        zero_pred = sum(1 for p in all_pred_texts if len(p) == 0)
        print(f"  wer={wer:.2f}%, avg_pred_len={total_pred_tokens/len(all_pred_texts):.1f}, "
              f"zero_pred={zero_pred}/{len(all_pred_texts)}")

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

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"训练完成。best_wer={best_wer:.2f}%")


if __name__ == "__main__":
    main()
