"""
CE-CSL 多模态 Dataset / DataLoader。
读取预处理的 .npy 关键点文件 + 视觉特征文件 + CSV 标签，返回 (T, D) tensor + token ids。
"""
import csv
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from activity_detect import segment_active_regions, extract_active_frames
from vocab_utils import clean_word
from augmentation import apply_augmentation


class KeypointDataset(Dataset):
    def __init__(self, base_dir, split, word2idx, activity_detect=False,
                 min_active_frames=5, gap_frames=3, visual_only=False,
                 no_visual=False, augment=False):
        self.base_dir = Path(base_dir)
        self.keypoint_dir = self.base_dir / "keypoints" / split
        self.visual_dir = self.base_dir / "visual_features" / split
        self.word2idx = word2idx
        self.activity_detect = activity_detect
        self.min_active_frames = min_active_frames
        self.gap_frames = gap_frames
        self.visual_only = visual_only
        self.no_visual = no_visual
        self.augment = augment
        self.samples = []  # [(video_id, token_ids, has_visual)]

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
                    has_visual = (self.visual_dir / f"{video_id}.npy").exists()
                    if self.visual_only and not has_visual:
                        continue
                    self.samples.append((video_id, token_ids, has_visual))

    def _gloss_to_ids(self, gloss):
        ids = []
        for w in gloss.split("/"):
            w = clean_word(w)
            if w and w in self.word2idx:
                ids.append(self.word2idx[w])
        return ids

    def __len__(self):
        return len(self.samples)

    def _load_visual(self, video_id):
        vf_path = self.visual_dir / f"{video_id}.npy"
        if vf_path.exists():
            return np.load(str(vf_path))  # (T, 576) float16
        return None

    def __getitem__(self, index):
        video_id, token_ids, _ = self.samples[index]
        kp_path = self.keypoint_dir / f"{video_id}.npy"
        kp = np.load(str(kp_path)).astype(np.float32)  # (T, 84)

        if self.no_visual:
            vf = None
        else:
            vf = self._load_visual(video_id)
            if vf is not None:
                if len(vf) != len(kp):
                    min_len = min(len(kp), len(vf))
                    kp = kp[:min_len]
                    vf = vf[:min_len]
            else:
                vf = np.zeros((len(kp), 576), dtype=np.float32)

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
                if not self.no_visual:
                    vf_seg = extract_active_frames(vf, segments)
                    vf = vf_seg

        if self.augment:
            kp = apply_augmentation(kp)

        if self.no_visual:
            features = kp
        else:
            features = np.concatenate([kp, vf.astype(np.float32)], axis=1)
        features = torch.from_numpy(features).float()
        return {
            "features": features,
            "label": torch.tensor(token_ids, dtype=torch.long),
            "video_id": video_id,
            "original_len": original_len,
            "trimmed": trimmed,
            "frames_removed": frames_removed,
        }


def collate_fn(batch):
    """按 T 降序排列，pad 到 batch_max_len。"""
    batch = sorted(batch, key=lambda x: len(x["features"]), reverse=True)

    features_list = [item["features"] for item in batch]
    labels = [item["label"] for item in batch]
    video_ids = [item["video_id"] for item in batch]

    input_lengths = torch.tensor([len(k) for k in features_list], dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    max_len = input_lengths[0]
    padded = []
    for k in features_list:
        T = k.shape[0]
        if T < max_len:
            pad = torch.zeros(max_len - T, k.shape[1])
            k = torch.cat([k, pad], dim=0)
        padded.append(k)

    padded = torch.stack(padded)  # (B, T_max, input_dim)
    padded = padded.permute(1, 0, 2)  # (T_max, B, input_dim)

    targets = torch.cat([l for l in labels], dim=0)  # (sum L_i,)

    return {
        "video": padded,
        "input_lengths": input_lengths,
        "label": targets,
        "target_lengths": target_lengths,
        "video_ids": video_ids,
    }


def collate_fn_grouped(batch, group_size=4):
    """
    按序列长度排序后分组 pad，每组 group_size 个样本。
    返回 list of dicts，每个 dict 与 collate_fn 格式一致。
    """
    batch = sorted(batch, key=lambda x: len(x["features"]))

    batches = []
    for i in range(0, len(batch), group_size):
        group = batch[i:i + group_size]
        if len(group) < 2:
            result = collate_fn(group)
            batches.append(result)
            continue

        features_list = [item["features"] for item in group]
        labels = [item["label"] for item in group]
        video_ids = [item["video_id"] for item in group]

        input_lengths = torch.tensor([len(k) for k in features_list], dtype=torch.long)
        target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

        max_len = input_lengths.max().item()
        padded = []
        for k in features_list:
            T = k.shape[0]
            if T < max_len:
                pad = torch.zeros(max_len - T, k.shape[1])
                k = torch.cat([k, pad], dim=0)
            padded.append(k)

        padded = torch.stack(padded)
        padded = padded.permute(1, 0, 2)  # (T_max, B, input_dim)
        targets = torch.cat([l for l in labels], dim=0)

        batches.append({
            "video": padded,
            "input_lengths": input_lengths,
            "label": targets,
            "target_lengths": target_lengths,
            "video_ids": video_ids,
        })

    return batches
