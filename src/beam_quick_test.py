"""Quick beam search test on 10 samples."""
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
from dataset import KeypointDataset
from decode import ctc_decode_batch


def seed_torch(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--beam", type=int, default=3)
    args = parser.parse_args()
    seed_torch(0)

    with open(BASE_DIR / "vocab_top478.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx2word = vocab["idx2word"]
    word2idx = vocab["word2idx"]
    blank = word2idx["<blank>"]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Checkpoint epoch={ckpt.get('epoch','?')}, wer={ckpt.get('wer','?')}")

    dev_set = KeypointDataset(KEYPOINT_BASE, "dev", word2idx, activity_detect=True,
                              min_active_frames=5, gap_frames=3,
                              visual_only=False, no_visual=True, augment=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLRModel(vocab_size=len(idx2word), input_dim=84, visual_fusion="none").to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Device: {device}, Beam: {args.beam}")

    N = 10
    for bw in [0, args.beam]:
        import time
        start = time.time()
        results = []
        with torch.no_grad():
            indices = list(range(N))
            for idx in tqdm(indices, desc=f"beam={bw}"):
                sample = dev_set[idx]
                feat = sample["features"].unsqueeze(1).to(device)
                input_lengths = torch.tensor([feat.shape[0]], dtype=torch.long, device=device)

                log_probs = model(feat, input_lengths)
                tokens = ctc_decode_batch(log_probs, blank, beam_width=bw)[0]

                label_tokens = sample["label"].tolist()
                pred_text = " ".join([idx2word[t] for t in tokens.tolist() if t < len(idx2word)])
                label_text = " ".join([idx2word[t] for t in label_tokens])
                results.append((pred_text, label_text))

        elapsed = time.time() - start
        print(f"\nbeam={bw}: {elapsed:.1f}s ({elapsed/N:.1f}s/sample)")
        for i, (p, l) in enumerate(results):
            print(f"  [{i}] pred: {p}")
            print(f"       ref:  {l}")
        print()

    from WER import WerList
    greedy_preds = [" ".join(results[i][0].split()) for i in range(N)]
    beam_preds = []
    with torch.no_grad():
        for idx in tqdm(range(N), desc=f"beam={args.beam}"):
            sample = dev_set[idx]
            feat = sample["features"].unsqueeze(1).to(device)
            input_lengths = torch.tensor([feat.shape[0]], dtype=torch.long, device=device)
            log_probs = model(feat, input_lengths)
            tokens = ctc_decode_batch(log_probs, blank, beam_width=args.beam)[0]
            beam_preds.append(" ".join([idx2word[t] for t in tokens.tolist() if t < len(idx2word)]))

    refs = [" ".join(results[i][1].split()) for i in range(N)]
    greedy_wer = WerList(hypotheses=greedy_preds, references=refs)["wer"]
    beam_wer = WerList(hypotheses=beam_preds, references=refs)["wer"]
    print(f"\n=== Results ({N} samples) ===")
    print(f"Greedy WER: {greedy_wer:.2f}%")
    print(f"Beam={args.beam} WER: {beam_wer:.2f}%")


if __name__ == "__main__":
    main()
