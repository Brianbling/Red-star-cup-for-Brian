"""
CE-CSL 原始视频 Dataset（用于端到端视觉特征 fine-tune）。
读取 .mp4 + CSV 标签，均匀采样 max_frames 帧，resize 224x224 + ImageNet 归一化。

返回 dict: {"features": (T, 3, 224, 224) float32, "label": token ids, "video_id": str}
短于 max_frames 的视频用零帧补齐（T 固定为 max_frames）。
"""
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from vocab_utils import clean_word

LETTER_DIRS = list("ABCDEFGHIJKL")

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def _find_video(video_dir, video_id):
    for letter in LETTER_DIRS:
        path = video_dir / letter / f"{video_id}.mp4"
        if path.exists():
            return path
    return None


def _uniform_frame_indices(total_frames, n):
    """均匀采样 n 个帧索引（含首尾），不重复。"""
    if total_frames <= n:
        return list(range(total_frames))
    return np.unique(
        np.linspace(0, total_frames - 1, n).round().astype(int)
    ).tolist()


class VideoDataset(Dataset):
    def __init__(self, video_base, label_base, split, word2idx, max_frames=90):
        self.video_dir = Path(video_base) / "video" / split
        self.word2idx = word2idx
        self.max_frames = max_frames
        self.samples = []  # [(video_id, video_path, token_ids)]

        label_path = Path(label_base) / "label" / f"{split}.csv"
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
                video_path = _find_video(self.video_dir, video_id)
                if video_path is None:
                    continue
                self.samples.append((video_id, video_path, token_ids))

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
        video_id, video_path, token_ids = self.samples[index]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return self._pad_frames(
                np.zeros((0, 224, 224, 3), dtype=np.uint8), token_ids, video_id
            )

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = _uniform_frame_indices(total, self.max_frames)

        frames = []
        pos_set = set(idxs)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i in pos_set:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if len(frames) == len(pos_set):
                    break
            i += 1
        cap.release()

        if len(frames) < len(idxs):
            # 视频提前结束（帧数统计与实际不符），重新按实读帧均匀采样
            frames = self._fallback_uniform(video_path)

        arr = np.stack(frames).astype(np.float32) if frames else np.zeros(
            (0, 224, 224, 3), dtype=np.float32
        )
        return self._pad_frames(arr, token_ids, video_id)

    def _fallback_uniform(self, video_path):
        """实际读到的帧不足：全量重读后均匀采样兜底。"""
        cap = cv2.VideoCapture(str(video_path))
        all_frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            all_frames.append(f)
        cap.release()
        if len(all_frames) <= self.max_frames:
            resized = [cv2.resize(f, (224, 224)) for f in all_frames]
            resized = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in resized]
            return resized
        idxs = _uniform_frame_indices(len(all_frames), self.max_frames)
        return [
            cv2.cvtColor(cv2.resize(all_frames[i], (224, 224)), cv2.COLOR_BGR2RGB)
            for i in idxs
        ]

    def _pad_frames(self, arr, token_ids, video_id):
        """arr: (T, 224, 224, 3) uint8/float32 → (max_frames, 3, 224, 224) float32，零填充。"""
        T = arr.shape[0]
        frames = torch.zeros((self.max_frames, 3, 224, 224), dtype=torch.float32)
        if T > 0:
            t = torch.from_numpy(arr[: min(T, self.max_frames)])
            # (n, H, W, 3) → (n, 3, H, W)
            t = t.permute(0, 3, 1, 2)
            t = t / 255.0
            for c in range(3):
                t[:, c] = (t[:, c] - NORMALIZE_MEAN[c]) / NORMALIZE_STD[c]
            frames[: t.shape[0]] = t
        return {
            "features": frames,  # (max_frames, 3, 224, 224)
            "label": torch.tensor(token_ids, dtype=torch.long),
            "video_id": video_id,
        }


def collate_fn(batch):
    """batch_size=1：直接取单样本，转成 (T, B, 3, 224, 224)。"""
    item = batch[0]
    return {
        "video": item["features"].unsqueeze(1),  # (T, 1, 3, 224, 224)
        "input_lengths": torch.tensor([item["features"].shape[0]], dtype=torch.long),
        "label": item["label"],
        "target_lengths": torch.tensor([len(item["label"])], dtype=torch.long),
        "video_ids": [item["video_id"]],
    }
