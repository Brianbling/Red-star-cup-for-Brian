"""
CE-CSL 关键点 Dataset / DataLoader。
读取预处理的 .npy 关键点文件 + CSV 标签，返回 (T, 84) tensor + token ids。
包含：坐标归一化（手部跨度）、clean_word 标签清洗、NaN 缺失语义。
"""
import csv
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

GLOBAL_FALLBACK_SCALE = 0.065
EPSILON = 1e-8


def _clean_word(word):
    """移除括号内容、首尾标点、尾部数字（与 build_vocab.py 保持一致）。"""
    chars = list(word)
    result = []
    skip = False
    for c in chars:
        if c in "(（[{（":
            skip = True
            continue
        if c in ")）]}）":
            skip = False
            continue
        if not skip:
            result.append(c)
    word = "".join(result).strip()
    if not word:
        return word
    if word[-1].isdigit() and not word[0].isdigit():
        word = word.rstrip("0123456789")
    word = word.strip()
    return word


def _compute_video_scale(kp_array):
    """
    计算每视频归一化 scale = 中位手部跨度（手腕 → 中指 MCP 距离）。
    kp_array: (T, 84) float32，NaN 表示缺失关键点，0.0 表示 legacy 零值。
    返回单一浮点数 scale。
    """
    spans = []
    for hand_offset in [0, 42]:  # 左手, 右手
        wrist = kp_array[:, hand_offset:hand_offset + 2]            # landmark 0
        middle_mcp = kp_array[:, hand_offset + 18:hand_offset + 20]  # landmark 9

        wrist_valid = ~np.all(wrist == 0, axis=1) & ~np.all(np.isnan(wrist), axis=1)
        mcp_valid = ~np.all(middle_mcp == 0, axis=1) & ~np.all(np.isnan(middle_mcp), axis=1)
        valid = wrist_valid & mcp_valid

        if valid.any():
            dists = np.linalg.norm(wrist[valid] - middle_mcp[valid], axis=1)
            spans.append(np.median(dists))

    if spans:
        return max(float(np.median(spans)), EPSILON)
    return GLOBAL_FALLBACK_SCALE


def _load_video_scales(keypoint_dir, split):
    """加载或计算 per-video scale 缓存。返回 {video_id: scale_float}。"""
    cache_path = Path(keypoint_dir) / f"{split}_scales.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    split_dir = Path(keypoint_dir) / split
    scales = {}
    for npy_path in sorted(split_dir.glob("*.npy")):
        video_id = npy_path.stem
        kp = np.load(str(npy_path))
        scales[video_id] = _compute_video_scale(kp)

    with open(cache_path, "w") as f:
        json.dump(scales, f)

    print(f"  Computed & cached scales for {len(scales)} videos -> {cache_path}")
    return scales


class KeypointDataset(Dataset):
    def __init__(self, base_dir, split, word2idx):
        self.base_dir = Path(base_dir)
        self.keypoint_dir = self.base_dir / "keypoints" / split
        self.word2idx = word2idx
        self.samples = []  # [(video_id, token_ids)]

        self.video_scales = _load_video_scales(
            self.base_dir / "keypoints", split
        )

        label_path = self.base_dir / "label" / f"{split}.csv"
        with open(label_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["Number"].strip()
                gloss = row["Gloss"].strip()
                if not gloss:
                    continue
                token_ids = self._gloss_to_ids(gloss)
                if len(token_ids) == 0:
                    continue
                if video_id == "train-01418":
                    continue
                npy_path = self.keypoint_dir / f"{video_id}.npy"
                if npy_path.exists():
                    self.samples.append((video_id, token_ids))

    def _gloss_to_ids(self, gloss):
        ids = []
        for w in gloss.split("/"):
            w = _clean_word(w)
            if w and w in self.word2idx:
                ids.append(self.word2idx[w])
        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_id, token_ids = self.samples[index]
        npy_path = self.keypoint_dir / f"{video_id}.npy"
        kp = np.load(str(npy_path))  # (T, 84), may contain NaN

        # Layer 2: NaN → 0.0（模型看到干净的零值）
        kp = np.nan_to_num(kp, nan=0.0)

        # 手腕平移 + 手部跨度归一化
        scale = self.video_scales.get(video_id, GLOBAL_FALLBACK_SCALE)
        kp = kp / scale

        kp = torch.from_numpy(kp).float()
        return {
            "keypoints": kp,          # (T, 84)
            "label": torch.tensor(token_ids, dtype=torch.long),  # (L,)
            "video_id": video_id,
        }


def collate_fn(batch):
    """按 T 降序排列，pad 到 batch_max_len。"""
    batch = sorted(batch, key=lambda x: len(x["keypoints"]), reverse=True)

    keypoints = [item["keypoints"] for item in batch]
    labels = [item["label"] for item in batch]
    video_ids = [item["video_id"] for item in batch]

    input_lengths = torch.tensor([len(k) for k in keypoints], dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    max_len = input_lengths[0]
    padded = []
    for k in keypoints:
        T = k.shape[0]
        if T < max_len:
            pad = torch.zeros(max_len - T, k.shape[1])
            k = torch.cat([k, pad], dim=0)
        padded.append(k)

    padded = torch.stack(padded)  # (B, T_max, 84)
    padded = padded.permute(1, 0, 2)  # (T_max, B, 84)  time-major for LSTM

    targets = torch.cat([l for l in labels], dim=0)  # (sum L_i,)

    return {
        "video": padded,
        "input_lengths": input_lengths,
        "label": targets,
        "target_lengths": target_lengths,
        "video_ids": video_ids,
    }
