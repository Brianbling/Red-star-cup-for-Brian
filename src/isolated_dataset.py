"""
孤立词数据集：把孤立词视频作为额外训练样本加入 CTC 主路。

孤立词样本是单 token 视频 → label = [token_idx]（单元素序列）。
与 KeypointDataset 返回相同 dict 格式，可共用 collate_fn。

用法:
  from isolated_dataset import IsolatedKeypointDataset, CombinedDataset

  csl_set = KeypointDataset(KEYPOINT_BASE, "train", word2idx, no_visual=True)
  iso_set = IsolatedKeypointDataset(INDEX_PATH, word2idx)
  combined = CombinedDataset(csl_set, iso_set)
  print(len(combined))   # 总样本数

注意:
  - 孤立词只有 84 维关键点，无视觉特征，须与 no_visual=True 的 CE-CSL 数据混合
  - 使用 isolated_words/index.json 定位孤立词 keypoint 文件
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_PATH = WORKTREE_ROOT / "isolated_words" / "index.json"


class IsolatedKeypointDataset(Dataset):
    """孤立词关键点数据集（CTC 训练用，单 token 标签）。"""

    def __init__(self, index_path, word2idx, activity_detect=False,
                 min_active_frames=5, gap_frames=3):
        """
        Args:
            index_path: isolated_words/index.json 路径
            word2idx: 词表 word2idx dict
            activity_detect: 是否裁剪手部丢失帧
        """
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        self.word2idx = word2idx
        self.activity_detect = activity_detect
        self.min_active_frames = min_active_frames
        self.gap_frames = gap_frames
        self.samples = []  # [(keypoint_path, token_idx)]

        for token, entries in index.items():
            if token not in word2idx:
                continue
            idx = word2idx[token]
            for e in entries:
                kp_path = Path(e["path"])
                if kp_path.exists():
                    self.samples.append((kp_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        kp_path, token_idx = self.samples[index]
        kp = np.load(str(kp_path)).astype(np.float32)  # (T, 84)

        original_len = len(kp)
        trimmed = False
        frames_removed = 0

        if self.activity_detect:
            from activity_detect import segment_active_regions, extract_active_frames
            segments = segment_active_regions(
                kp, self.min_active_frames, self.gap_frames)
            if len(segments) > 0:
                kp_seg = extract_active_frames(kp, segments)
                trimmed = len(kp_seg) < original_len
                frames_removed = original_len - len(kp_seg)
                kp = kp_seg

        return {
            "features": torch.from_numpy(kp).float(),
            "label": torch.tensor([token_idx], dtype=torch.long),
            "video_id": kp_path.stem,
            "original_len": original_len,
            "trimmed": trimmed,
            "frames_removed": frames_removed,
        }


class CombinedDataset(Dataset):
    """把多个数据集顺序拼接为一个 Dataset（用于 CE-CSL train + 孤立词）。"""

    def __init__(self, *datasets):
        self.datasets = list(datasets)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, index):
        for d in self.datasets:
            if index < len(d):
                return d[index]
            index -= len(d)
        raise IndexError(index)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(WORKTREE_ROOT / "src"))
    from dataset import KeypointDataset

    vocab_path = Path("D:/red star project") / "vocab_top478.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)["word2idx"]

    csl_set = KeypointDataset(
        Path("E:/CE-CSL/CE-CSL"), "train", word2idx, no_visual=True)
    iso_set = IsolatedKeypointDataset(DEFAULT_INDEX_PATH, word2idx)
    combined = CombinedDataset(csl_set, iso_set)

    print(f"CE-CSL train: {len(csl_set)}")
    print(f"孤立词: {len(iso_set)}")
    print(f"合并: {len(combined)}")

    if len(iso_set) > 0:
        s = iso_set[0]
        print(f"孤立词样本: features={tuple(s['features'].shape)}, "
              f"label={s['label'].tolist()}, video_id={s['video_id']}")

    # 验证 CombinedDataset 索引边界
    print(f"combined[0]: {combined[0]['video_id']}")
    print(f"combined[len-1]: {combined[len(combined)-1]['video_id']}")
