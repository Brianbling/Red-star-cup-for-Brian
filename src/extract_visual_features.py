"""
Phase 1b: Visual feature extraction
CE-CSL video → MobileNetV3-Small features (every frame) → .npy

Output: (T, 576) float16  — same frame count as keypoints
"""
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import argparse
import csv
import time
from pathlib import Path
import torch
from torchvision import models, transforms

BASE_DIR = Path("D:/red star project")
VIDEO_DIR = BASE_DIR / "CE-CSL/CE-CSL/video"
LABEL_DIR = BASE_DIR / "CE-CSL/CE-CSL/label"
FEATURE_DIR = BASE_DIR / "CE-CSL/CE-CSL/visual_features"

LETTER_DIRS = list("ABCDEFGHIJKL")

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def find_video(split, video_id):
    for letter in LETTER_DIRS:
        path = VIDEO_DIR / split / letter / f"{video_id}.mp4"
        if path.exists():
            return path
    return None


def load_label_map(split):
    csv_path = LABEL_DIR / f"{split}.csv"
    label_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["Number"].strip()
            label_map[video_id] = row["Gloss"].strip()
    return label_map


def extract_visual_features(extractor, video_path, device, batch_size=32):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).float().to(device)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor / 255.0
        tensor = NORMALIZE(tensor)
        frames.append(tensor)

    cap.release()

    if len(frames) == 0:
        return None

    all_features = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = torch.stack(frames[i:i + batch_size])
            feats = extractor(batch)
            feats = feats.mean(dim=[-2, -1])
            all_features.append(feats.cpu().half())

    return torch.cat(all_features, dim=0).numpy()


def run_split(split, max_videos, extractor, device):
    label_map = load_label_map(split)
    save_dir = FEATURE_DIR / split
    save_dir.mkdir(parents=True, exist_ok=True)

    total = len(label_map)
    processed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Processing {split}: {total} videos"
          + (f" (limit {max_videos})" if max_videos else ""))
    print(f"{'='*60}")

    count = 0
    for video_id in label_map:
        if max_videos and processed >= max_videos:
            break

        count += 1
        save_path = save_dir / f"{video_id}.npy"

        if save_path.exists():
            skipped += 1
            continue

        video_path = find_video(split, video_id)
        if video_path is None:
            if video_id == "train-01418":
                print(f"  [SKIP] {video_id}: video missing (known)")
            else:
                print(f"  [WARN] {video_id}: video missing")
            errors += 1
            continue

        features = extract_visual_features(extractor, video_path, device)
        if features is None:
            print(f"  [ERROR] {video_id}: no frames")
            errors += 1
            continue

        np.save(str(save_path), features)
        processed += 1

        elapsed = time.time() - start_time
        eta = (elapsed / processed) * (
            min(total, max_videos or total) - processed - skipped
        ) if processed > 0 else 0
        print(f"  [{count}/{total}] {video_id}: {features.shape[0]}f "
              f"| ETA: {eta/60:.1f}min")

    print(f"\n{split} done: {processed} processed, {skipped} skipped, "
          f"{errors} errors")
    return processed, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="CE-CSL visual feature extraction (MobileNetV3-Small)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = models.mobilenet_v3_small(weights='DEFAULT').to(device)
    model.eval()
    extractor = model.features
    print("Model: MobileNetV3-Small, feature dim: 576")

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for split in splits:
        p, s, e = run_split(split, args.max_videos, extractor, device)
        total_processed += p
        total_skipped += s
        total_errors += e

    print(f"\n{'='*60}")
    print(f"All done: {total_processed} processed, {total_skipped} skipped, "
          f"{total_errors} errors")
    print(f"Output: {FEATURE_DIR}")


if __name__ == "__main__":
    main()
