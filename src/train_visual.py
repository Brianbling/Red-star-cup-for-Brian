"""
视觉端到端模型训练脚本（MobileNetV3-Large fine-tune）。
用法: python train_visual.py --vocab vocab_top478.json --max-frames 90 --epochs 1

与 train.py 相同的 CTC loss + blank_penalty + entropy + blank_bias=5.32。
backbone 用较低 LR（1e-4），head 用 1e-3。
"""
import json
import sys
import argparse
import functools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

BASE_DIR = Path("D:/red star project")
WORKTREE_DIR = Path(__file__).resolve().parent.parent
VIDEO_BASE = Path("E:/CE-CSL/CE-CSL")
LABEL_BASE = Path("E:/CE-CSL/CE-CSL")
DEFAULT_CHECKPOINT_DIR = WORKTREE_DIR / "checkpoints" / "visual"

sys.path.insert(0, str(WORKTREE_DIR / "src"))
sys.path.insert(1, str(BASE_DIR / "TFNet-main"))

from model_visual import VisualSLRModel
from video_dataset import VideoDataset, collate_fn
from decode import ctc_decode_batch


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


def compute_diagnostics(model, dataloader, device, blank=0, beam_width=0):
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

                p_blank_on_valid = (pred == blank).float().mean().item()

                decoded = ctc_decode_batch(lp.unsqueeze(1), blank, beam_width)
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
    parser.add_argument("--max-frames", type=int, default=90,
                        help="Uniformly sampled frames per video")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch)")
    parser.add_argument("--freeze-stages", type=int, default=3,
                        help="Freeze first N backbone stages (0..4)")
    parser.add_argument("--backbone-lr", type=float, default=1e-4,
                        help="Learning rate for backbone params")
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="Learning rate for head (conv/lstm/fc) params")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--blank-bias", type=float, default=5.32,
                        help="Initial FC bias for blank token")
    parser.add_argument("--backbone-chunk", type=int, default=30,
                        help="Frames per backbone forward pass")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint save directory")
    args = parser.parse_args()

    seed_torch(0)

    with open(BASE_DIR / args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    vocab_size = len(idx2word)
    blank = word2idx["<blank>"]

    print(f"词表: {args.vocab}, 大小: {vocab_size}, blank={blank}")
    print(f"max_frames={args.max_frames}, grad_accum={args.grad_accum}, "
          f"freeze_stages={args.freeze_stages}, backbone_lr={args.backbone_lr}, "
          f"head_lr={args.head_lr}, AMP={args.amp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_set = VideoDataset(VIDEO_BASE, LABEL_BASE, "train", word2idx,
                             max_frames=args.max_frames)
    dev_set = VideoDataset(VIDEO_BASE, LABEL_BASE, "dev", word2idx,
                           max_frames=args.max_frames)
    print(f"训练集: {len(train_set)} 样本, 验证集: {len(dev_set)} 样本")

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    model = VisualSLRModel(
        vocab_size=vocab_size,
        blank_bias=args.blank_bias,
        backbone_chunk=args.backbone_chunk,
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone"):
            continue
        head_params.append(p)

    print(f"backbone 可训练参数: {sum(p.numel() for p in backbone_trainable):,}")
    print(f"head 可训练参数: {sum(p.numel() for p in head_params):,}")

    param_groups = [
        {"params": backbone_trainable, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ]
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    ctc_loss_fn = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=False)

    blank_penalty_weight = 20.0
    blank_threshold = 0.55
    entropy_weight = 0.01

    diag = compute_diagnostics(model, dev_loader, device, blank)
    print(f"初始化诊断: P(blank)={diag['p_blank']:.3f}, "
          f"zero_pred={diag['zero_pred_pct']:.1f}%, "
          f"collapse={diag['collapsed_pct']:.1f}%")

    best_wer = float("inf")
    patience_counter = 0
    patience = 15

    for epoch in range(args.epochs):
        model.train()
        losses = []
        optimizer.zero_grad()

        for step, item in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            video = item["video"].to(device)
            input_lengths = item["input_lengths"].to(device)
            label = item["label"].to(device)
            target_lengths = item["target_lengths"].to(device)

            with torch.autocast("cuda", enabled=args.amp):
                log_probs = model(video, input_lengths)
                ctc_input_lengths = (input_lengths // 4).clamp(min=1)
                ctc_loss = ctc_loss_fn(log_probs, label, ctc_input_lengths, target_lengths)

                probs = torch.exp(log_probs)
                valid_mask = torch.arange(probs.shape[0], device=probs.device).unsqueeze(1) < ctc_input_lengths.unsqueeze(0)
                valid_blank = probs[:, :, blank][valid_mask]
                blank_prob = valid_blank.mean()
                blank_penalty = (blank_prob - blank_threshold).clamp(min=0)

                entropy = -(probs * log_probs).sum(dim=-1).mean()

                loss = ctc_loss + blank_penalty_weight * blank_penalty + entropy_weight * entropy

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                optimizer.zero_grad()

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
              f"lr={optimizer.param_groups[0]['lr']:.6f}/{optimizer.param_groups[1]['lr']:.6f}")

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

                pred_tokens = ctc_decode_batch(log_probs, blank)
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

        torch.save(checkpoint, checkpoint_dir / "last.pt")

        if wer < best_wer:
            best_wer = wer
            patience_counter = 0
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            print(f"  [NEW BEST] wer={wer:.2f}% → best.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"训练完成。best_wer={best_wer:.2f}%")


if __name__ == "__main__":
    main()
