"""
Phase 1: 孤立词分类数据加载。
读取预处理的 .npy 关键点文件 + 单标签，返回 (T, 84) tensor + class id。
支持 SLR（单标签）和 CE-CSL（Gloss 按 / 分割取第一个词作孤立词标签）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict


class IsolatedWordDataset(Dataset):
    """Isolated word classification dataset.

    Labels are single word class indices. Samples are (T, 84) keypoint sequences.
    Optional activity detection trims hand-loss frames.
    """

    def __init__(self, keypoint_dir, label_map, word2idx, labels_json=None,
                 activity_detect=False, min_active_frames=5, gap_frames=3):
        """
        Args:
            keypoint_dir: directory with .npy files
            label_map: dict {video_id: word_str} or {video_id: [gloss_list]}
            word2idx: dict mapping word strings to class indices
            labels_json: optional path to SLR labels.json for word_id -> word lookup
            activity_detect: trim hand-loss frames
        """
        self.keypoint_dir = Path(keypoint_dir)
        self.word2idx = word2idx
        self.activity_detect = activity_detect
        self.min_active_frames = min_active_frames
        self.gap_frames = gap_frames
        self.samples = []  # [(video_id, word_idx)]

        unk_idx = word2idx.get("<unk>", len(word2idx))

        for video_id, label in label_map.items():
            npy_path = self.keypoint_dir / f"{video_id}.npy"
            if not npy_path.exists():
                continue

            if isinstance(label, list):
                word = label[0] if label else None
            else:
                word = label

            if word is None:
                continue

            if word in word2idx:
                self.samples.append((video_id, word2idx[word]))
            elif unk_idx is not None:
                self.samples.append((video_id, unk_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, word_idx = self.samples[idx]
        npy_path = self.keypoint_dir / f"{video_id}.npy"
        kp = np.load(str(npy_path)).astype(np.float32)  # (T, 84)

        original_len = len(kp)

        if self.activity_detect:
            from activity_detect import segment_active_regions, extract_active_frames
            segments = segment_active_regions(kp, self.min_active_frames, self.gap_frames)
            if len(segments) > 0:
                kp = extract_active_frames(kp, segments)

        return {
            "features": torch.from_numpy(kp).float(),
            "label": torch.tensor(word_idx, dtype=torch.long),
            "video_id": video_id,
            "original_len": original_len,
        }


def collate_isolated(batch):
    """Pad to batch_max_len, return (T_max, B, 84) time-major tensor + labels."""
    batch = sorted(batch, key=lambda x: len(x["features"]), reverse=True)

    features_list = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    input_lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)
    max_len = input_lengths[0]

    padded = []
    for f in features_list:
        T = f.shape[0]
        if T < max_len:
            pad = torch.zeros(max_len - T, f.shape[1])
            f = torch.cat([f, pad], dim=0)
        padded.append(f)

    padded = torch.stack(padded)  # (B, T_max, 84)
    padded = padded.permute(1, 0, 2)  # (T_max, B, 84)

    return {
        "features": padded,
        "input_lengths": input_lengths,
        "label": labels,
    }


def load_slr_labels(labels_json, word_map_json):
    """Load SLR isolated word labels from JSON files.

    Args:
        labels_json: path to labels.json ({word_id -> word_str})
        word_map_json: path to word_map.json (optional, for re-mapping)

    Returns:
        label_map: dict {video_id: word_str}
    """
    import json
    with open(labels_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # raw is {word_id: word_str}
    label_map = {}
    for wid, word in raw.items():
        label_map[wid] = word
    return label_map
