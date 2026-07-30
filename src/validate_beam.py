"""Load baseline best.pt and evaluate with beam search on dev set only."""
import json, sys, argparse, torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path("D:/red star project")
WORKTREE_DIR = Path(__file__).resolve().parent.parent
KEYPOINT_BASE = Path("E:/CE-CSL/CE-CSL")

sys.path.insert(0, str((WORKTREE_DIR / "src").resolve()))
sys.path.insert(1, str(BASE_DIR / "TFNet-main"))

from model import SLRModel
from dataset import KeypointDataset, collate_fn
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--vocab", type=str, default="vocab_top478.json")
    args = parser.parse_args()

    seed_torch(0)

    with open(BASE_DIR / args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx2word = vocab["idx2word"]
    word2idx = vocab["word2idx"]
    blank = word2idx["<blank>"]

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  epoch={ckpt.get('epoch','?')}, wer={ckpt.get('wer','?')}")

    dev_set = KeypointDataset(
        KEYPOINT_BASE, "dev", word2idx,
        activity_detect=True, min_active_frames=5, gap_frames=3,
        visual_only=False, no_visual=True, augment=False,
    )
    print(f"Dev samples: {len(dev_set)}")

    from torch.utils.data import DataLoader
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 84
    model = SLRModel(vocab_size=len(idx2word), input_dim=input_dim,
                     visual_fusion="none").to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Device: {device}")
    print(f"Beam width: {args.beam_width}")

    all_pred_texts = []
    all_label_texts = []
    zero_pred = 0
    total_blank = 0
    total_frames = 0

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Validating"):
            video = batch["video"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            label = batch["label"]
            target_lengths = batch["target_lengths"]

            log_probs = model(video, input_lengths)
            probs = torch.exp(log_probs)

            ctc_input_lengths = (input_lengths // 4).clamp(min=1)
            for b in range(video.shape[1]):
                T = ctc_input_lengths[b].item()
                if T == 0:
                    continue
                pred = probs[:T, b, :].argmax(dim=-1)
                total_blank += (pred == blank).sum().item()
                total_frames += T

            pred_tokens = ctc_decode_batch(log_probs, blank, args.beam_width)
            for b in range(video.shape[1]):
                bt = pred_tokens[b].tolist()
                pred_text = [idx2word[int(t)] for t in bt if int(t) < len(idx2word)]
                label_start = sum(target_lengths[:b])
                label_end = label_start + target_lengths[b]
                label_tokens = label[label_start:label_end].tolist()
                label_text = [idx2word[t] for t in label_tokens]
                all_pred_texts.append(pred_text)
                all_label_texts.append(label_text)
                if len(pred_text) == 0:
                    zero_pred += 1

    wer_result = compute_wer(all_pred_texts, all_label_texts)
    print(f"\n=== Exp-3 (beam_width={args.beam_width}) Results ===")
    print(f"WER={wer_result['wer']:.2f}%")
    print(f"S={wer_result['sub_rate']:.1f}%, D={wer_result['del_rate']:.1f}%, I={wer_result['ins_rate']:.1f}%")
    print(f"P(blank)={total_blank/total_frames if total_frames > 0 else 0:.3f}")
    print(f"Zero pred: {zero_pred}/{len(all_pred_texts)} ({zero_pred/len(all_pred_texts)*100:.1f}%)")


if __name__ == "__main__":
    main()
