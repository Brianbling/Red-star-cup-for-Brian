"""
主路模型训练脚本。
用法: python train.py                                    # 全量词表(3515)
       python train.py --vocab vocab_top478.json          # 子词表快速验证
       python train.py --activity-detect                  # 启用 activity detection
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
WORKTREE_DIR = Path(__file__).resolve().parent.parent
KEYPOINT_BASE = BASE_DIR / "CE-CSL/CE-CSL"
CHECKPOINT_DIR = WORKTREE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WORKTREE_DIR / "src"))
sys.path.insert(1, str(BASE_DIR / "TFNet-main"))

from model import SLRModel
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
    return WerList(hypotheses=hypotheses, references=references)


def compute_diagnostics(model, dataloader, device, blank=0):
    """Per-sample blank ratio and zero_pred rate on valid frames."""
    model.eval()
    total_frames = 0
    blank_frames = 0
    zero_pred_count = 0
    total_samples = 0
    per_sample_p_blank = []

    with torch.no_grad():
        for batch in dataloader:
            video = batch["video"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            log_probs = model(video, input_lengths)
            ctc_input_lengths = (input_lengths // 4).clamp(min=1)

            for b in range(video.shape[1]):
                T = ctc_input_lengths[b].item()
                if T == 0:
                    continue
                lp = log_probs[:T, b, :]
                pred = lp.argmax(dim=-1)

                blank_frames += (pred == blank).sum().item()
                total_frames += T

                # Only count on valid (non-zero) frames
                p_blank_on_valid = (pred == blank).float().mean().item()

                decoded = ctc_greedy_decode(lp.unsqueeze(1), blank)
                if len(decoded[0]) == 0:
                    zero_pred_count += 1

                per_sample_p_blank.append(p_blank_on_valid)
                total_samples += 1

    if total_samples == 0:
        return {"p_blank": 0, "zero_pred_pct": 0, "collapsed_pct": 0}

    return {
        "p_blank": blank_frames / total_frames if total_frames > 0 else 0,
        "zero_pred_pct": zero_pred_count / total_samples * 100,
        "collapsed_pct": sum(1 for p in per_sample_p_blank if p >= 0.99) / total_samples * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, default="vocab.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--activity-detect", action="store_true", default=False,
                        help="Enable activity detection to trim hand-loss frames")
    parser.add_argument("--min-active-frames", type=int, default=5,
                        help="Min consecutive active frames for a segment")
    parser.add_argument("--gap-frames", type=int, default=3,
                        help="Max gap of zero frames to merge segments")
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
    print(f"activity_detect: {args.activity_detect}, "
          f"min_active_frames={args.min_active_frames}, gap_frames={args.gap_frames}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    train_set = KeypointDataset(
        KEYPOINT_BASE, "train", word2idx,
        activity_detect=args.activity_detect,
        min_active_frames=args.min_active_frames,
        gap_frames=args.gap_frames,
    )
    dev_set = KeypointDataset(
        KEYPOINT_BASE, "dev", word2idx,
        activity_detect=args.activity_detect,
        min_active_frames=args.min_active_frames,
        gap_frames=args.gap_frames,
    )
    print(f"训练集: {len(train_set)} 样本, 验证集: {len(dev_set)} 样本")

    if args.activity_detect:
        trimmed_train = sum(1 for i in range(len(train_set))
                            if train_set[i]["trimmed"])
        trimmed_dev = sum(1 for i in range(len(dev_set))
                          if dev_set[i]["trimmed"])
        all_frames_removed_train = sum(train_set[i]["frames_removed"]
                                       for i in range(len(train_set)))
        all_frames_removed_dev = sum(dev_set[i]["frames_removed"]
                                     for i in range(len(dev_set)))
        all_original_frames_train = sum(train_set[i]["original_len"]
                                        for i in range(len(train_set)))
        all_original_frames_dev = sum(dev_set[i]["original_len"]
                                      for i in range(len(dev_set)))
        print(f"  activity detect: train {trimmed_train}/{len(train_set)} 样本被切分, "
              f"移除 {all_frames_removed_train}/{all_original_frames_train} 帧 "
              f"({all_frames_removed_train/max(all_original_frames_train,1)*100:.1f}%)")
        print(f"  activity detect: dev {trimmed_dev}/{len(dev_set)} 样本被切分, "
              f"移除 {all_frames_removed_dev}/{all_original_frames_dev} 帧 "
              f"({all_frames_removed_dev/max(all_original_frames_dev,1)*100:.1f}%)")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=collate_fn,
    )

    model = SLRModel(vocab_size=vocab_size).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # Print init diagnostics
    diag = compute_diagnostics(model, dev_loader, device, blank)
    print(f"初始化诊断: P(blank)={diag['p_blank']:.3f}, "
          f"zero_pred={diag['zero_pred_pct']:.1f}%, "
          f"collapse={diag['collapsed_pct']:.1f}%")

    ctc_loss_fn = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    blank_penalty_weight = 20.0
    entropy_weight = 0.01

    epochs = 100
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
                ctc_input_lengths = (input_lengths // 4).clamp(min=1)
                ctc_loss = ctc_loss_fn(log_probs, label, ctc_input_lengths, target_lengths)

                probs = torch.exp(log_probs)
                blank_prob = probs[:, :, blank].mean()
                blank_penalty = (blank_prob - 0.85).clamp(min=0)

                entropy = -(probs * log_probs).sum(dim=-1).mean()

                loss = ctc_loss + blank_penalty_weight * blank_penalty + entropy_weight * entropy

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

            losses.append({"ctc": ctc_loss.item(), "bp": blank_penalty.item(),
                           "entropy": entropy.item(), "total": loss.item()})

        if not losses:
            print(f"Epoch {epoch+1}: 所有 loss 为 inf/nan，跳过")
            continue

        avg_ctc = np.mean([l["ctc"] for l in losses])
        avg_bp = np.mean([l["bp"] for l in losses])
        avg_entropy = np.mean([l["entropy"] for l in losses])
        avg_total = np.mean([l["total"] for l in losses])
        print(f"Epoch {epoch+1}: ctc={avg_ctc:.4f}, bp={avg_bp:.4f}, "
              f"ent={avg_entropy:.4f}, total={avg_total:.4f}, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

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
                    bt = pred_tokens[b].tolist()
                    pred_text = [idx2word[int(t)] for t in bt if int(t) < len(idx2word)]
                    label_start = sum(target_lengths[:b])
                    label_end = label_start + target_lengths[b]
                    label_tokens = label[label_start:label_end].tolist()
                    label_text = [idx2word[t] for t in label_tokens]
                    all_pred_texts.append(pred_text)
                    all_label_texts.append(label_text)

        wer_result = compute_wer(all_pred_texts, all_label_texts)
        wer = wer_result["wer"]
        scheduler.step(wer)

        diag = compute_diagnostics(model, dev_loader, device, blank)
        zero_pred = sum(1 for p in all_pred_texts if len(p) == 0)
        print(f"  wer={wer:.2f}%, S={wer_result['sub_rate']:.1f}%, "
              f"D={wer_result['del_rate']:.1f}%, I={wer_result['ins_rate']:.1f}%, "
              f"P(blank)={diag['p_blank']:.3f}, "
              f"collapse={diag['collapsed_pct']:.1f}%, "
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
