"""
Phase 1: 关键点预处理（多进程版）
CE-CSL 视频 → MediaPipe Hands 逐帧提取关键点 → .npy

每个 worker 进程持有独立的 HandLandmarker，通过 Pool.imap_unordered 并行。

用法:
  python src/preprocess_keypoints.py --split train --max_videos 10   # 小批量验证
  python src/preprocess_keypoints.py --split train                    # 全量单进程
  python src/preprocess_keypoints.py --split all --workers 4          # 4 进程并行
  python src/preprocess_keypoints.py --split all                      # 默认全核心并行
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
from multiprocessing import Pool, cpu_count

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

# --- 每个 worker 进程的全局 landmarker ---
_worker_landmarker = None


def _worker_init():
    """Pool initializer：每个 worker 进程启动时创建自己的 HandLandmarker"""
    global _worker_landmarker
    _worker_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
    )


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


def process_video(landmarker, video_path):
    """逐帧处理一个视频，返回 (npy_array_or_None, error_message_or_None)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, f"无法打开视频: {video_path}"

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extract_keypoints(landmarker, frame))

    cap.release()

    if len(frames) == 0:
        return None, f"视频无帧: {video_path}"

    return np.stack(frames, axis=0), None


def _worker_process(task):
    """Pool worker：处理单个视频，保存 .npy，返回状态元组（不传大数组回主进程）"""
    video_id, video_path, save_path = task
    global _worker_landmarker

    sequence, error = process_video(_worker_landmarker, video_path)
    if error:
        return (video_id, 0, error)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), sequence)
    return (video_id, len(sequence), None)


def load_label_map(split):
    csv_path = LABEL_DIR / f"{split}.csv"
    label_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["Number"].strip()
            label_map[video_id] = row["Gloss"].strip()
    return label_map


def main():
    parser = argparse.ArgumentParser(description="CE-CSL 关键点预处理（多进程）")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--max_videos", type=int, default=None,
                        help="每个 split 最多处理 N 个视频")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"进程数（默认 CPU 核心数 = {cpu_count()}）")
    args = parser.parse_args()

    workers = args.workers or cpu_count()
    print(f"使用 {workers} 个进程并行")
    print(f"模型: {MODEL_PATH}")

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for split in splits:
        label_map = load_label_map(split)
        save_dir = KEYPOINT_DIR / split
        save_dir.mkdir(parents=True, exist_ok=True)

        # 构建任务列表
        tasks = []
        skipped = 0
        errors = 0

        for video_id in label_map:
            if args.max_videos and len(tasks) >= args.max_videos:
                break

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

            tasks.append((video_id, str(video_path), save_path))

        total_tasks = len(tasks)
        print(f"\n{'='*60}")
        print(f"处理 {split}: {len(label_map)} 个视频, "
              f"{total_tasks} 待处理, {skipped} 跳过, {errors} 错误"
              + (f" (限制前 {args.max_videos} 个)" if args.max_videos else ""))
        print(f"{'='*60}")

        if total_tasks == 0:
            continue

        start_time = time.time()
        processed = 0
        worker_errors = 0
        total_frames = 0

        with Pool(processes=workers, initializer=_worker_init) as pool:
            for video_id, n_frames, error_msg in pool.imap_unordered(
                _worker_process, tasks
            ):
                processed += 1
                total_frames += n_frames

                if error_msg:
                    worker_errors += 1
                    print(f"  [{processed}/{total_tasks}] {video_id}: [ERROR] {error_msg}")
                    continue

                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - processed) / rate if rate > 0 else 0
                print(f"  [{processed}/{total_tasks}] {video_id}: {n_frames}帧 "
                      f"| {rate:.2f}视频/秒 | ETA: {eta/60:.1f}min")

        elapsed = time.time() - start_time
        ok = processed - worker_errors
        total_processed += ok
        total_skipped += skipped
        total_errors += errors + worker_errors

        print(f"\n{split} 完成: {ok} 处理, {skipped} 跳过, {errors + worker_errors} 错误")
        print(f"耗时: {elapsed/60:.1f}min, 总帧数: {total_frames}, "
              f"平均: {total_frames/(elapsed or 1):.1f}帧/秒")

    print(f"\n{'='*60}")
    print(f"全部完成: {total_processed} 处理, {total_skipped} 跳过, {total_errors} 错误")
    print(f"输出目录: {KEYPOINT_DIR}")


if __name__ == "__main__":
    main()
