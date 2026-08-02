"""
归一化已有的全部 .npy 关键点文件（CE-CSL）。

每帧每只手独立处理：
  wrist   = landmark 0（hand[0:2]）
  lm9     = 中指 MCP（hand[18:20]）
  scale   = ||lm9 - wrist||_2（< 0.02 时用 0.02 兜底）
  坐标     = (coord - wrist) / scale
  手部完全丢失（该手 42 值全零）→ 保持全零

用法:
  python src/normalize_existing.py
  python src/normalize_existing.py --splits train --workers 8
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SOURCE_BASE = Path("E:/CE-CSL/CE-CSL/keypoints")
OUTPUT_BASE = Path("E:/CE-CSL/CE-CSL/keypoints_normalized")
EPSILON = 1e-12          # 帧级全零判定（该手未检测）
MIN_HAND_SCALE = 0.02    # 归一化尺度下限：1e-6 会把 1e-6~0.02 的小手放大 12~650 倍（audit 确认）


def normalize_hand(hand):
    """wrist-centering + 中指根缩放。hand: (42,) float32。返回 (归一化后的 42, scale)。"""
    wrist = hand[0:2]
    lm9 = hand[18:20]
    scale = float(np.hypot(lm9[0] - wrist[0], lm9[1] - wrist[1]))
    if scale < MIN_HAND_SCALE:
        scale = MIN_HAND_SCALE
    out = np.zeros_like(hand)
    for j in range(21):
        out[j * 2] = (hand[j * 2] - wrist[0]) / scale
        out[j * 2 + 1] = (hand[j * 2 + 1] - wrist[1]) / scale
    return out, scale


def normalize_file(src_path, dst_path):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    arr = np.load(str(src_path)).astype(np.float32)  # (T, 84)
    out = arr.copy()
    scales = []
    n_zero_scale = 0
    n_valid_frames = 0

    for t in range(arr.shape[0]):
        frame = arr[t]
        for offset in (0, 42):
            hand = frame[offset:offset + 42]
            if np.abs(hand).sum() < 1e-12:
                continue  # 该手未检测到 → 保持全零
            hand_n, scale = normalize_hand(hand)
            out[t, offset:offset + 42] = hand_n
            scales.append(scale)
            n_valid_frames += 1
            if scale < MIN_HAND_SCALE:
                n_zero_scale += 1

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(dst_path), out)
    return {
        "name": src_path.name,
        "n_frames": int(arr.shape[0]),
        "n_valid_frames": n_valid_frames,
        "n_zero_scale": n_zero_scale,
        "scales": np.asarray(scales, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=str(SOURCE_BASE))
    parser.add_argument("--output", type=str, default=str(OUTPUT_BASE))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--splits", type=str, default="train,dev,test")
    args = parser.parse_args()

    source_base = Path(args.source)
    output_base = Path(args.output)
    splits = args.splits.split(",")

    all_scales = []
    all_results = []

    for split in splits:
        src_dir = source_base / split
        dst_dir = output_base / split
        files = sorted(src_dir.glob("*.npy"))
        if len(files) == 0:
            print(f"[{split}] 无文件，跳过")
            continue
        print(f"[{split}] {len(files)} 个文件开始处理")

        tasks = [(f, dst_dir / f.name) for f in files]
        with Pool(args.workers) as pool:
            for i, res in enumerate(pool.starmap(normalize_file, tasks), 1):
                all_results.append(res)
                if i % 1000 == 0:
                    print(f"  [{split}] {i}/{len(files)}")
        print(f"[{split}] 完成")

    if not all_results:
        print("未处理任何文件")
        return

    for r in all_results:
        all_scales.append(r["scales"])
    scales = np.concatenate(all_scales) if all_scales else np.array([])

    n_zero_scale_files = sum(1 for r in all_results if r["n_zero_scale"] > 0)
    n_valid_total = sum(r["n_valid_frames"] for r in all_results)
    n_frames_total = sum(r["n_frames"] for r in all_results)

    print(f"\n{'='*60}")
    print(f"处理文件数: {len(all_results)}")
    print(f"总帧数: {n_frames_total}, 有效手帧(手部被检测): {n_valid_total}")
    if len(scales) > 0:
        print(f"scale 统计（当前数据上计算，wrist→中指MCP 距离）:")
        print(f"  mean={scales.mean():.6f}, median={np.median(scales):.6f}")
        print(f"  min={scales.min():.6f}, max={scales.max():.6f}")
        print(f"  接近 1.0 的比例 (|scale-1|<0.02): {np.abs(scales-1.0).mean() if len(scales) else 0:.6f}")
    print(f"含 zero-scale 帧的文件数: {n_zero_scale_files}")
    print(f"输出目录: {output_base}")

    # 历史尺度参考（如果存在）
    hist = {}
    for split in splits:
        json_path = source_base / f"{split}_scales.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            vals = np.array(list(d.values()), dtype=np.float64)
            hist[split] = {
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n": int(len(vals)),
            }
    if hist:
        print(f"\n历史 pre-normalization 尺度 (来自 {split}_scales.json):")
        for split, s in hist.items():
            print(f"  {split}: n={s['n']}, mean={s['mean']:.5f}, "
                  f"median={s['median']:.5f}, min={s['min']:.5f}, max={s['max']:.5f}")


if __name__ == "__main__":
    main()
