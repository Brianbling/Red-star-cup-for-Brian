"""
孤立词 → CE-CSL 词表 索引构建。
扫描 isolated_words/keypoints/{basic,common} 下的 .npy，
用 clean_word(文件名 stem) 匹配 vocab.json 的 word2idx（默认与 train.py --vocab 对齐）。

输出: isolated_words/index.json
  { "<token>": [{"dataset": "basic", "path": "...", "video_id": "..."}], ... }

用法:
  python build_isolated_index.py [--vocab PATH]
"""

import argparse
import json
import sys
from pathlib import Path

MAIN_BASE = Path("D:/red star project")
KEYPOINT_ROOT = MAIN_BASE / "isolated_words" / "keypoints"
INDEX_PATH = MAIN_BASE / "isolated_words" / "index.json"
VOCAB_PATH = MAIN_BASE / "vocab.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab_utils import clean_word


def main():
    global VOCAB_PATH
    parser = argparse.ArgumentParser(description="构建孤立词 → 词表索引")
    parser.add_argument("--vocab", default=str(VOCAB_PATH), help="词表 json 路径（默认 vocab.json，与 train.py --vocab 对齐）")
    args = parser.parse_args()
    VOCAB_PATH = Path(args.vocab)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    word2idx = vocab["word2idx"]

    index = {}
    total_videos = 0
    for dataset in ["basic", "common"]:
        kp_dir = KEYPOINT_ROOT / dataset
        if not kp_dir.exists():
            print(f"[WARN] {kp_dir} 不存在，跳过")
            continue
        for npy_path in sorted(kp_dir.glob("*.npy")):
            raw = npy_path.stem
            token = clean_word(raw)
            if token in word2idx:
                index.setdefault(token, []).append({
                    "dataset": dataset,
                    "path": str(npy_path),
                    "video_id": raw,
                })
                total_videos += 1

    # 排序保证输出稳定
    for token in index:
        index[token].sort(key=lambda e: (e["dataset"], e["video_id"]))

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    covered_tokens = len(index)
    matched_videos = total_videos
    per_dataset = {}
    for token, entries in index.items():
        for e in entries:
            per_dataset[e["dataset"]] = per_dataset.get(e["dataset"], 0) + 1

    print(f"词表大小: {len(word2idx)}")
    print(f"覆盖 token 数: {covered_tokens}")
    print(f"匹配视频总数: {matched_videos}")
    for ds, cnt in sorted(per_dataset.items()):
        print(f"  {ds}: {cnt}")
    print(f"索引写入: {INDEX_PATH}")

    # 示例条目
    if index:
        sample_token = next(iter(index))
        print(f"\n示例条目: {sample_token} -> {index[sample_token]}")


if __name__ == "__main__":
    main()
