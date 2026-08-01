"""
CSL-Daily 关键点数据集：把 CSL-Daily 连续句关键点混入 CE-CSL 训练。

CSL-Daily 关键点（E:/CSL-Daily/CSL-Daily/keypoints/{split}/）由本项目
MediaPipe 管线抽取，(T, 84) float32，与 CE-CSL 格式逐字节一致。
标签在 keypoints/labels.json：{video_id: [gloss, ...]}（gloss 列表，非 "/" 分隔）。

返回 dict 格式与 KeypointDataset 一致，可直接复用 collate_fn / CombinedDataset。
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from activity_detect import segment_active_regions, extract_active_frames
from vocab_utils import clean_word


class CSLDailyDataset(Dataset):
    def __init__(self, base_dir, split, word2idx, activity_detect=False,
                 min_active_frames=5, gap_frames=3):
        self.base_dir = Path(base_dir)
        self.keypoint_dir = self.base_dir / "keypoints" / split
        self.word2idx = word2idx
        self.activity_detect = activity_detect
        self.min_active_frames = min_active_frames
        self.gap_frames = gap_frames

        with open(self.base_dir / "keypoints" / "labels.json", "r", encoding="utf-8") as f:
            labels = json.load(f)

        self.samples = []
        for npy_path in self.keypoint_dir.glob("*.npy"):
            video_id = npy_path.stem
            gloss_list = labels.get(video_id)
            if not gloss_list:
                continue
            token_ids = self._gloss_to_ids(gloss_list)
            if len(token_ids) == 0:
                continue
            self.samples.append((npy_path, token_ids))

    def _gloss_to_ids(self, gloss_list):
        ids = []
        for w in gloss_list:
            w = clean_word(w)
            if w and w in self.word2idx:
                ids.append(self.word2idx[w])
        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        npy_path, token_ids = self.samples[index]
        kp = np.load(str(npy_path)).astype(np.float32)  # (T, 84)

        original_len = len(kp)
        trimmed = False
        frames_removed = 0

        if self.activity_detect:
            segments = segment_active_regions(
                kp, self.min_active_frames, self.gap_frames)
            if len(segments) > 0:
                kp_seg = extract_active_frames(kp, segments)
                trimmed = len(kp_seg) < original_len
                frames_removed = original_len - len(kp_seg)
                kp = kp_seg

        return {
            "features": torch.from_numpy(kp).float(),
            "label": torch.tensor(token_ids, dtype=torch.long),
            "video_id": npy_path.stem,
            "original_len": original_len,
            "trimmed": trimmed,
            "frames_removed": frames_removed,
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    vocab_path = Path("D:/red star project") / "vocab_top478.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)["word2idx"]

    for split in ["train", "dev"]:
        ds = CSLDailyDataset(
            Path("E:/CSL-Daily/CSL-Daily"), split, word2idx, activity_detect=True)
        print(f"CSL-Daily {split}: {len(ds)} 样本")
        if len(ds) > 0:
            s = ds[0]
            print(f"  样本: features={tuple(s['features'].shape)}, "
                  f"label={s['label'].tolist()}, video_id={s['video_id']}")
