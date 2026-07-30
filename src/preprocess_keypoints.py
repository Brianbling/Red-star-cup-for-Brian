"""
Phase 1: 关键点预处理（多进程 + 坐标归一化 + worker 级 landmarker 复用）
CE-CSL 视频 → MediaPipe Hands 逐帧提取关键点 → .npy

用法:
  python preprocess_keypoints.py --split train --max_videos 10   # 小批量验证
  python preprocess_keypoints.py --split all --workers 6          # 全量
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
from multiprocessing import Pool

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

BASE_DIR = Path("D:/red star project")
VIDEO_DIR = BASE_DIR / "CE-CSL/CE-CSL/video"
LABEL_DIR = BASE_DIR / "CE-CSL/CE-CSL/label"
KEYPOINT_DIR = BASE_DIR / "CE-CSL/CE-CSL/keypoints"
MODEL_PATH = str(BASE_DIR / "models/hand_landmarker.task")

PRESENCE_THRESHOLD = 0.6
LETTER_DIRS = list("ABCDEFGHIJKL")

# worker 局部 landmarker，由 initializer 设置
_worker_landmarker = None


def _init_worker():
    global _worker_landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    _worker_landmarker = HandLandmarker.create_from_options(options)


def _find_video(split, video_id):
    for letter in LETTER_DIRS:
        path = VIDEO_DIR / split / letter / f"{video_id}.mp4"
        if path.exists():
            return str(path)
    return None


def _normalize_hand(coords_2d):
    """以手腕(点0)为原点，腕→中指根(点9)距离为分母。若关键特征缺失则返回零。"""
    wrist = coords_2d[0].copy()
    mcp9 = coords_2d[9].copy()
    if np.linalg.norm(wrist) < 1e-6 or np.linalg.norm(mcp9) < 1e-6:
        return np.zeros(42, dtype=np.float32)
    scale = np.linalg.norm(mcp9 - wrist)
    if scale < 1e-4:
        return np.zeros(42, dtype=np.float32)
    centered = coords_2d - wrist
    return (centered / scale).astype(np.float32).reshape(42)


def _extract_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    result = _worker_landmarker.detect(mp_image)

    left_raw = np.zeros((21, 2), dtype=np.float32)
    right_raw = np.zeros((21, 2), dtype=np.float32)
    left_valid = False
    right_valid = False

    if not result.hand_landmarks:
        return np.zeros(84, dtype=np.float32)

    for i, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name
        target = left_raw if handedness == "Left" else right_raw

        for j, lm in enumerate(landmarks):
            presence = lm.presence if lm.presence is not None else 1.0
            if presence > PRESENCE_THRESHOLD:
                target[j, 0] = lm.x
                target[j, 1] = lm.y

        if handedness == "Left":
            left_valid = True
        else:
            right_valid = True

    kp = np.zeros(84, dtype=np.float32)
    if left_valid:
        kp[:42] = _normalize_hand(left_raw)
    if right_valid:
        kp[42:] = _normalize_hand(right_raw)
    return kp


def process_one_video(args):
    split, video_id, save_path = args
    video_path = _find_video(split, video_id)
    if video_path is None:
        return (video_id, 0, "video_missing")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return (video_id, 0, "cant_open")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(_extract_frame(frame))

        cap.release()

        if len(frames) == 0:
            return (video_id, 0, "no_frames")

        sequence = np.stack(frames, axis=0)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, sequence)
        return (video_id, len(frames), "ok")
    except Exception as e:
        return (video_id, 0, f"error:{e}")


def build_task_list(split, max_videos):
    csv_path = LABEL_DIR / f"{split}.csv"
    label_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["Number"].strip()
            label_map[video_id] = row["Gloss"].strip()

    save_dir = KEYPOINT_DIR / split
    save_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    skipped = 0
    for video_id in label_map:
        if max_videos and len(tasks) >= max_videos:
            break
        save_path = save_dir / f"{video_id}.npy"
        if save_path.exists():
            skipped += 1
            continue
        tasks.append((split, video_id, str(save_path)))
    return tasks, skipped


def run_split(split, max_videos, workers):
    tasks, pre_skipped = build_task_list(split, max_videos)
    total = pre_skipped + len(tasks)

    print(f"\n{'='*60}")
    print(f"Split={split}: {total} 视频 ({len(tasks)} 待处理, {pre_skipped} 跳过), workers={workers}")
    print(f"{'='*60}")

    if not tasks:
        return 0, pre_skipped, 0

    processed = 0
    errors = 0
    start_time = time.time()

    with Pool(workers, initializer=_init_worker) as pool:
        for video_id, n_frames, status in pool.imap_unordered(process_one_video, tasks):
            if status == "ok":
                processed += 1
            else:
                errors += 1
                if status == "video_missing":
                    if video_id != "train-01418":
                        print(f"  [WARN] {video_id}: 视频不存在")
                else:
                    print(f"  [ERROR] {video_id}: {status}")

            done = processed + errors
            if done % 200 == 0 or done == len(tasks):
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate / 60 if rate > 0 else 0
                print(f"  [{done}/{len(tasks)}] ok={processed} err={errors} "
                      f"{rate:.1f}vid/min ETA={eta:.1f}min")

    print(f"\n{split} 完成: {processed} ok, {pre_skipped} skip, {errors} err")
    return processed, pre_skipped, errors


def main():
    parser = argparse.ArgumentParser(description="CE-CSL 关键点预处理（多进程 + 坐标归一化）")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    print(f"模型: {MODEL_PATH}")
    print(f"归一化: 手腕原点 + 中指MCP缩放 | presence>{PRESENCE_THRESHOLD}")

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    tp, ts, te = 0, 0, 0
    for split in splits:
        p, s, e = run_split(split, args.max_videos, args.workers)
        tp += p
        ts += s
        te += e
    print(f"\n全完成: {tp} ok, {ts} skip, {te} err → {KEYPOINT_DIR}")


if __name__ == "__main__":
    main()
