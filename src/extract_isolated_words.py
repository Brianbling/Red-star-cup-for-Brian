"""
孤立词关键点提取。
CSL_basic_dataset / CSL_common_dataset 视频 → MediaPipe Hands → (T, 84) float32 .npy

与 E:/CE-CSL/CE-CSL 主数据对齐：
  - MediaPipe IMAGE 模式（VIDEO 模式有遥测超时问题）
  - 手腕归一化：normalize_hand（(p - wrist) / ||middle_mcp - wrist||_2，每只手独立）
    E:/CE-CSL 关键点已按此公式归一化（middle_mcp norm == 1.0），孤立词必须同分布
  - 左手 0-41，右手 42-83，presence < 0.6 置零，手部丢失填全零

用法:
  python extract_isolated_words.py --dataset all --limit 50   # 子集验证
  python extract_isolated_words.py --dataset all              # 全量
"""

import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import multiprocessing
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

# 主仓库（数据源 + 模型文件 + 输出）
MAIN_BASE = Path("D:/red star project")
OUTPUT_DIR = MAIN_BASE / "isolated_words"
MODEL_PATH = MAIN_BASE / "models/hand_landmarker.task"

DATASETS = {
    "basic": MAIN_BASE / "CSL_basic_dataset",
    "common": MAIN_BASE / "CSL_common_dataset",
}

PRESENCE_THRESHOLD = 0.6
HAND_SCALE_FALLBACK = 0.1
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def normalize_hand(kp_hand):
    """单只手(42维)手腕居中 + 手尺寸归一化，与 fix-preprocess-normalization 一致。"""
    wrist = kp_hand[WRIST_IDX * 2: WRIST_IDX * 2 + 2]
    middle_mcp = kp_hand[MIDDLE_MCP_IDX * 2: MIDDLE_MCP_IDX * 2 + 2]
    if (wrist == 0).all():
        return kp_hand
    hand_scale = np.linalg.norm(middle_mcp - wrist)
    if hand_scale < 1e-6:
        hand_scale = HAND_SCALE_FALLBACK
    normalized = np.zeros_like(kp_hand)
    for j in range(21):
        px, py = kp_hand[j * 2], kp_hand[j * 2 + 1]
        normalized[j * 2] = (px - wrist[0]) / hand_scale
        normalized[j * 2 + 1] = (py - wrist[1]) / hand_scale
    return normalized


def create_landmarker(model_path=None):
    path = Path(model_path) if model_path else MODEL_PATH
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def extract_keypoints(landmarker, frame_bgr):
    """单帧 → 84 维（左手 0-41 + 右手 42-83，各含 21 点 × 2）。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    left_kp = np.zeros(42, dtype=np.float32)
    right_kp = np.zeros(42, dtype=np.float32)

    if result.hand_landmarks:
        for i, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name
            target = left_kp if handedness == "Left" else right_kp
            for j, lm in enumerate(landmarks):
                presence = lm.presence if lm.presence is not None else 1.0
                if presence > PRESENCE_THRESHOLD:
                    target[j * 2] = lm.x
                    target[j * 2 + 1] = lm.y

    left_kp = normalize_hand(left_kp)
    right_kp = normalize_hand(right_kp)
    return np.concatenate([left_kp, right_kp])


def open_video(video_path):
    """打开视频。直接路径失败时回退到复制到 ASCII 临时路径（防中文文件名编码问题）。"""
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap, None
    cap.release()

    tmp_path = Path(tempfile.gettempdir()) / "isolated_tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)
    ascii_name = f"tmp_{abs(hash(str(video_path))) % (10 ** 8)}.mp4"
    ascii_full = tmp_path / ascii_name
    try:
        shutil.copy2(str(video_path), str(ascii_full))
        cap = cv2.VideoCapture(str(ascii_full))
        if cap.isOpened():
            return cap, ascii_full
        cap.release()
    except Exception as e:
        print(f"  [WARN] 复制到临时路径失败 {video_path}: {e}")
    return None, None


def process_video(landmarker, video_path):
    cap, tmp_handle = open_video(video_path)
    if cap is None:
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extract_keypoints(landmarker, frame))
    cap.release()

    if tmp_handle is not None:
        try:
            tmp_handle.unlink()
        except OSError:
            pass

    if len(frames) == 0:
        return None
    return np.stack(frames, axis=0)


def _init_worker(model_path):
    """每个 worker 进程初始化一次 landmarker（避免每次视频重复加载模型）。"""
    global _WORKER_LANDMARKER
    _WORKER_LANDMARKER = create_landmarker(model_path)


def _worker(video_path, save_dir):
    """单视频 worker（multiprocessing 用）：复用进程级 landmarker + 断点续跑。"""
    video_path = Path(video_path)
    save_path = Path(save_dir) / f"{video_path.stem}.npy"
    if save_path.exists():
        return "skipped", video_path.stem, int(np.load(str(save_path)).shape[0])

    landmarker = _WORKER_LANDMARKER
    seq = process_video(landmarker, video_path)

    if seq is None:
        return "error", video_path.stem, 0
    np.save(str(save_path), seq)
    return "ok", video_path.stem, len(seq)


def run_dataset(dataset, limit, num_workers, model_path):
    video_dir = DATASETS[dataset]
    save_dir = OUTPUT_DIR / "keypoints" / dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(p for p in video_dir.glob("*.mp4"))
    if limit:
        mp4s = mp4s[:limit]

    print(f"\n{'='*60}")
    print(f"处理 {dataset}: {len(mp4s)} 个视频, {num_workers} workers")
    print(f"{'='*60}")

    start = time.time()
    processed = skipped = errors = 0
    frames_ranges = []

    worker_fn = partial(_worker, save_dir=str(save_dir))

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(str(model_path),),
    ) as pool:
        for status, video_id, n_frames in pool.imap_unordered(worker_fn, mp4s):
            if status == "ok":
                processed += 1
                frames_ranges.append(n_frames)
            elif status == "skipped":
                skipped += 1
                frames_ranges.append(n_frames)
            else:
                errors += 1

            done = processed + skipped + errors
            if done % 50 == 0 or done == len(mp4s):
                elapsed = time.time() - start
                eta = (elapsed / done) * (len(mp4s) - done) if done > 0 else 0
                print(f"  [{done}/{len(mp4s)}] ok={processed} skip={skipped} "
                      f"err={errors} | ETA: {eta/60:.1f}min")

    print(f"\n{dataset} 完成: {processed} 处理, {skipped} 跳过, {errors} 错误")
    if frames_ranges:
        print(f"  帧数范围: min={min(frames_ranges)}, max={max(frames_ranges)}, "
              f"mean={np.mean(frames_ranges):.1f}")
    return processed, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="孤立词关键点提取")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["basic", "common", "all"])
    parser.add_argument("--limit", type=int, default=None,
                        help="每个数据集最多处理 N 个视频（子集验证用）")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="多进程 worker 数（8 在 16GB 内存下安全）")
    args = parser.parse_args()

    print(f"模型: {MODEL_PATH}")
    print(f"输出: {OUTPUT_DIR}")

    datasets = ["basic", "common"] if args.dataset == "all" else [args.dataset]
    total_processed = total_skipped = total_errors = 0

    for ds in datasets:
        p, s, e = run_dataset(ds, args.limit, args.num_workers, MODEL_PATH)
        total_processed += p
        total_skipped += s
        total_errors += e

    print(f"\n全部完成: {total_processed} 处理, {total_skipped} 跳过, {total_errors} 错误")


if __name__ == "__main__":
    main()
