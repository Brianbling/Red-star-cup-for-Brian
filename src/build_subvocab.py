"""
构建 top-N 子词表用于快速管线验证。
用法: python build_subvocab.py --min_freq 10
输出: vocab_top{N}.json
"""
import json
import csv
import argparse
from collections import Counter
from pathlib import Path
from dataset import _clean_word

BASE_DIR = Path("D:/red star project")
LABEL_DIR = BASE_DIR / "CE-CSL/CE-CSL/label"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=int, default=10)
    args = parser.parse_args()

    word_counts = Counter()
    for split in ["train", "dev", "test"]:
        csv_path = LABEL_DIR / f"{split}.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gloss = row["Gloss"].strip()
                if not gloss:
                    continue
                for w in gloss.split("/"):
                    w = _clean_word(w)
                    if w and w not in (".", "。", "，", ","):
                        word_counts[w] += 1

    top_words = sorted(
        [w for w, c in word_counts.items() if c >= args.min_freq],
        key=lambda w: -word_counts[w],
    )

    idx2word = ["<blank>"] + top_words
    word2idx = {w: i for i, w in enumerate(idx2word)}

    total_tokens = sum(word_counts.values())
    covered = sum(word_counts[w] for w in top_words)

    print(f"min_freq >= {args.min_freq}: {len(top_words)} words")
    print(f"Token coverage: {covered}/{total_tokens} ({100*covered/total_tokens:.1f}%)")

    out_path = BASE_DIR / f"vocab_top{len(top_words)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"word2idx": word2idx, "idx2word": idx2word}, f,
                  ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
