"""
构建 CE-CSL 词汇表。
用法: python build_vocab.py
输出: vocab.json  {"<blank>": 0, "词1": 1, ...}
"""
import csv
import json
from pathlib import Path

BASE_DIR = Path("D:/red star project")
LABEL_DIR = BASE_DIR / "CE-CSL/CE-CSL/label"

from vocab_utils import clean_word


def main():
    word_set = set()

    for split in ["train", "dev", "test"]:
        csv_path = LABEL_DIR / f"{split}.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gloss = row["Gloss"].strip()
                if not gloss:
                    continue
                for w in gloss.split("/"):
                    w = clean_word(w)
                    if w and w != "." and w != "。" and w != "，" and w != ",":
                        word_set.add(w)

    idx2word = ["<blank>"] + sorted(word_set)
    word2idx = {w: i for i, w in enumerate(idx2word)}

    vocab_path = BASE_DIR / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"word2idx": word2idx, "idx2word": idx2word}, f, ensure_ascii=False, indent=2)

    print(f"词汇表大小: {len(idx2word)} (含 blank)")
    print(f"已保存: {vocab_path}")


if __name__ == "__main__":
    main()
