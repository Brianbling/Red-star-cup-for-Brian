"""
孤立词分类预训练脚本。
在 SLR 25K 孤立词上训练手势分类能力，产出的 Conformer encoder 权重可直接加载到 CTC 模型。

用法:
  python pretrain_isolated.py --vocab vocab.json --data-dir SLR_Dataset/keypoints
"""

import json
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path("D:/red star project")
WORKTREE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = WORKTREE_DIR / "checkpoints" / "pretrain_isolated"

sys.path.insert(0, str(WORKTREE_DIR / "src"))

from model import ConformerEncoder, ClassificationHead
from dataset_isolated import IsolatedWordDataset, collate_isolated


def seed_torch(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_accuracy(encoder, classifier, dataloader, device):
    encoder.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["label"].to(device)
            enc_out, _ = encoder(features, input_lengths)
            logits = classifier(enc_out)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0


def compute_per_class_accuracy(encoder, classifier, dataloader, device, num_classes, top_n=10):
    """Return per-class counts and per-class accuracy for monitoring."""
    encoder.eval()
    classifier.eval()
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["label"].to(device)
            enc_out, _ = encoder(features, input_lengths)
            logits = classifier(enc_out)
            pred = logits.argmax(dim=-1)
            for c in range(num_classes):
                mask = labels == c
                class_total[c] += mask.sum()
                class_correct[c] += (pred[mask] == c).sum()

    acc_per_class = []
    for c in range(num_classes):
        if class_total[c] > 0:
            acc_per_class.append((c, class_correct[c].item() / class_total[c].item(), class_total[c].item()))

    acc_per_class.sort(key=lambda x: x[2], reverse=True)
    return acc_per_class


def main():
    ap = argparse.ArgumentParser(description="Isolated word pretraining")
    ap.add_argument("--vocab", type=str, default="vocab.json")
    ap.add_argument("--data-dir", type=str, default="SLR_Dataset/keypoints")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--pooling", type=str, default="attention",
                    choices=["attention", "mean"])
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--activity-detect", action="store_true", default=False)
    ap.add_argument("--checkpoint-dir", type=str, default=None)
    args = ap.parse_args()

    seed_torch(0)

    with open(BASE_DIR / args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    num_classes = len(idx2word)
    blank = word2idx.get("<blank>", 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"词表: {args.vocab}, 类别数: {num_classes}")
    print(f"Pooling: {args.pooling}, Label smoothing: {args.label_smoothing}")

    data_dir = BASE_DIR / args.data_dir

    train_dir = data_dir / "train"
    dev_dir = data_dir / "dev"

    label_map_train = {}
    label_map_dev = {}

    if (data_dir / "labels.json").exists():
        import json as j
        with open(data_dir / "labels.json", "r", encoding="utf-8") as f:
            raw = j.load(f)
        for vid, word in raw.items():
            npy_path = train_dir / f"{vid}.npy"
            if npy_path.exists():
                label_map_train[vid] = word
            npy_path = dev_dir / f"{vid}.npy"
            if npy_path.exists():
                label_map_dev[vid] = word
    else:
        for npy in train_dir.glob("*.npy"):
            vid = npy.stem
            label_map_train[vid] = vid.split("_")[0] if "_" in vid else vid
        for npy in dev_dir.glob("*.npy"):
            vid = npy.stem
            label_map_dev[vid] = vid.split("_")[0] if "_" in vid else vid

    train_set = IsolatedWordDataset(
        train_dir, label_map_train, word2idx,
        activity_detect=args.activity_detect,
    )
    dev_set = IsolatedWordDataset(
        dev_dir, label_map_dev, word2idx,
        activity_detect=args.activity_detect,
    )

    print(f"训练集: {len(train_set)}, 验证集: {len(dev_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_isolated,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_isolated,
    )

    encoder = ConformerEncoder(input_dim=84, d_model=256, num_heads=4,
                               num_blocks=4, dropout=0.3)
    encoder = encoder.to(device)
    classifier = ClassificationHead(d_model=256, num_classes=num_classes,
                                    pooling=args.pooling).to(device)

    print(f"参数量: {sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in classifier.parameters()):,}")

    ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    all_params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        encoder.train()
        classifier.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast("cuda", enabled=args.amp):
                enc_out, new_lens = encoder(features, input_lengths)
                logits = classifier(enc_out)
                loss = ce_loss(logits, labels)

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                optimizer.step()

            losses.append(loss.item())

        if not losses:
            continue

        avg_loss = np.mean(losses)
        train_acc = compute_accuracy(encoder, classifier, train_loader, device)
        dev_acc = compute_accuracy(encoder, classifier, dev_loader, device)
        scheduler.step(dev_acc)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, "
              f"train_acc={train_acc:.2%}, "
              f"dev_acc={dev_acc:.2%}, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % 5 == 0:
            per_class = compute_per_class_accuracy(encoder, classifier, dev_loader, device, num_classes)
            top10 = per_class[:10]
            bottom10 = per_class[-10:] if len(per_class) >= 10 else per_class
            print(f"  Top-10 (most frequent): "
                  f"{[(idx2word[str(c)], f'{a:.1%}') for c, a, _ in top10]}")
            if len(per_class) >= 20:
                print(f"  Bottom-10 (least frequent): "
                      f"{[(idx2word[str(c)], f'{a:.1%}') for c, a, _ in bottom10]}")

        checkpoint = {
            "epoch": epoch,
            "dev_acc": dev_acc,
            "encoder_state_dict": encoder.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": num_classes,
            "idx2word": idx2word,
            "word2idx": word2idx,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")

        if dev_acc > best_acc:
            best_acc = dev_acc
            patience_counter = 0
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            print(f"  [NEW BEST] dev_acc={dev_acc:.2%}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"预训练完成: best_dev_acc={best_acc:.2%}")


if __name__ == "__main__":
    main()
