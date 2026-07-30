"""
CE-CSL 关键点 Dataset / DataLoader。
读取预处理的 .npy 关键点文件 + CSV 标签，返回 (T, 84) tensor + token ids。
"""
import csv
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from activity_detect import segment_active_regions, extract_active_frames
from vocab_utils import clean_word


class KeypointDataset(Dataset):
    def __init__(self, base_dir, split, word2idx, activity_detect=False,
                 min_active_frames=5, gap_frames=3):
        self.base_dir = Path(base_dir)
        self.keypoint_dir = self.base_dir / "keypoints" / split
        self.word2idx = word2idx
        self.activity_detect = activity_detect
        self.min_active_frames = min_active_frames
        self.gap_frames = gap_frames
        self.samples = []  # [(video_id, token_ids)]

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
            w = clean_word(w)
            if w and w in self.word2idx:
                ids.append(self.word2idx[w])
        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_id, token_ids = self.samples[index]
        npy_path = self.keypoint_dir / f"{video_id}.npy"
        kp = np.load(str(npy_path))  # (T, 84)

        original_len = len(kp)

        if self.activity_detect:
            segments = segment_active_regions(
                kp, self.min_active_frames, self.gap_frames)
            kp = extract_active_frames(kp, segments)
            trimmed = len(segments) > 0 and len(kp) < original_len
            frames_removed = original_len - len(kp) if trimmed else 0
        else:
            trimmed = False
            frames_removed = 0

        kp = torch.from_numpy(kp).float()
        return {
            "keypoints": kp,
            "label": torch.tensor(token_ids, dtype=torch.long),
            "video_id": video_id,
            "original_len": original_len,
            "trimmed": trimmed,
            "frames_removed": frames_removed,
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
