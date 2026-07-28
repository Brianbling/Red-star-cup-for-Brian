"""
Phase 1: 关键点预处理
CE-CSL 视频 → MediaPipe Hands 逐帧提取关键点 → .npy

API: MediaPipe 0.10.35 (tasks API, HandLandmarker, IMAGE mode)
输出格式: (T, 84) float32
  左手 21 点 (x,y) + 右手 21 点 (x,y) = 42 点 × 2 = 84 维
  presence < 0.6 的坐标置零
  手部完全丢失时填全零

用法:
  python preprocess_keypoints.py --split train --max_videos 10   # 小批量验证
  python preprocess_keypoints.py --split train                    # 全量
  python preprocess_keypoints.py --split all                     # 全部
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

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

# --- 路径配置 ---
BASE_DIR = Path("D:/red star project")
VIDEO_DIR = BASE_DIR / "CE-CSL/CE-CSL/video"
LABEL_DIR = BASE_DIR / "CE-CSL/CE-CSL/label"
KEYPOINT_DIR = BASE_DIR / "CE-CSL/CE-CSL/keypoints"
MODEL_PATH = BASE_DIR / "models/hand_landmarker.task"

PRESENCE_THRESHOLD = 0.6
LETTER_DIRS = list("ABCDEFGHIJKL")


def create_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def find_video(split, video_id):
    for letter in LETTER_DIRS:
        path = VIDEO_DIR / split / letter / f"{video_id}.mp4"
        if path.exists():
            return path
    return None


def extract_keypoints(landmarker, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    keypoints = np.zeros(84, dtype=np.float32)

    if not result.hand_landmarks:
        return keypoints

    for i, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name
        offset = 0 if handedness == "Left" else 42

        for j, lm in enumerate(landmarks):
            presence = lm.presence if lm.presence is not None else 1.0
            if presence > PRESENCE_THRESHOLD:
                keypoints[offset + j * 2] = lm.x
                keypoints[offset + j * 2 + 1] = lm.y

    return keypoints


def process_video(landmarker, video_path, save_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 无法打开视频: {video_path}")
        return 0, False

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp = extract_keypoints(landmarker, frame)
        frames.append(kp)

    cap.release()

    if len(frames) == 0:
        print(f"  [ERROR] 视频无帧: {video_path}")
        return 0, False

    sequence = np.stack(frames, axis=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), sequence)
    return len(frames), True


def load_label_map(split):
    csv_path = LABEL_DIR / f"{split}.csv"
    label_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["Number"].strip()
            label_map[video_id] = row["Gloss"].strip()
    return label_map


def run_split(split, max_videos, landmarker):
    label_map = load_label_map(split)
    save_dir = KEYPOINT_DIR / split
    save_dir.mkdir(parents=True, exist_ok=True)

    total = len(label_map)
    processed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"处理 {split}: {total} 个视频" + (f" (限制前 {max_videos} 个)" if max_videos else ""))
    print(f"{'='*60}")

    count = 0
    for video_id, gloss in label_map.items():
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
                print(f"  [SKIP] {video_id}: 视频不存在（已知孤立行）")
            else:
                print(f"  [WARN] {video_id}: 视频不存在")
            errors += 1
            continue

        n_frames, ok = process_video(landmarker, video_path, save_path)
        processed += 1

        elapsed = time.time() - start_time
        eta = (elapsed / processed) * (min(total, max_videos or total) - processed - skipped) if processed > 0 else 0
        print(f"  [{count}/{total}] {video_id}: {n_frames}帧 "
              f"| gloss: {gloss[:25]} | ETA: {eta/60:.1f}min")

    print(f"\n{split} 完成: {processed} 处理, {skipped} 跳过(已存在), {errors} 错误")
    return processed, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="CE-CSL 关键点预处理")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--max_videos", type=int, default=None,
                        help="每个 split 最多处理 N 个视频（小批量验证用）")
    args = parser.parse_args()

    print(f"加载模型: {MODEL_PATH}")
    landmarker = create_landmarker()

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for split in splits:
        p, s, e = run_split(split, args.max_videos, landmarker)
        total_processed += p
        total_skipped += s
        total_errors += e

    landmarker.close()

    print(f"\n{'='*60}")
    print(f"全部完成: {total_processed} 处理, {total_skipped} 跳过, {total_errors} 错误")
    print(f"输出目录: {KEYPOINT_DIR}")


if __name__ == "__main__":
    main()
