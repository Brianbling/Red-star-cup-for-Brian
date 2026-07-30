"""
Phase 1: 关键点预处理 — 多数据集支持
CE-CSL / SLR / CSL-Daily 视频 → MediaPipe Hands 逐帧提取关键点 → .npy

API: MediaPipe 0.10.35 (tasks API, HandLandmarker, IMAGE mode)
输出格式: (T, 84) float32，左手 21 点 (x,y) + 右手 21 点 (x,y)
  presence < 0.6 的坐标置零，手部完全丢失时填全零
  手腕归一化：(point - wrist) / ||middle_mcp - wrist||_2，每只手独立

用法:
  python preprocess_keypoints.py --dataset ce-csl --split train --max_videos 10
  python preprocess_keypoints.py --dataset slr --max_videos 10
  python preprocess_keypoints.py --dataset csl-daily --split train --max_videos 10
"""

import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import argparse
import csv
import pickle
import time
import json
from pathlib import Path
from collections import defaultdict

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

PRESENCE_THRESHOLD = 0.6
HAND_SCALE_FALLBACK = 0.1
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def normalize_hand(kp_hand):
    """Wrist-centering + hand-scale normalization for one hand (21 points, 42 values)."""
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


def extract_keypoints(landmarker, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    left_kp = np.zeros(42, dtype=np.float32)
    right_kp = np.zeros(42, dtype=np.float32)

    if not result.hand_landmarks:
        return np.concatenate([left_kp, right_kp])

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


def create_landmarker(model_path):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def process_video(landmarker, video_path, save_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
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
        return 0, False

    sequence = np.stack(frames, axis=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), sequence)
    return len(frames), True


# ─── Dataset label loaders ─────────────────────────────────────────────────

def load_ce_csl_labels(label_dir, split):
    csv_path = Path(label_dir) / f"{split}.csv"
    label_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["Number"].strip()] = row["Gloss"].strip()
    return label_map


def load_csl_daily_labels(label_dir):
    pkl_path = Path(label_dir) / "csl2020ct_v2.pkl"
    split_path = Path(label_dir) / "split_1.txt"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    video_info = {}
    for info in data["info"]:
        video_info[info["name"]] = info["label_gloss"]

    video_split = {}
    with open(split_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                video_split[parts[0]] = parts[1]

    return video_info, video_split


# ─── Video finders ─────────────────────────────────────────────────────────

CE_CSL_LETTER_DIRS = list("ABCDEFGHIJKL")


def find_ce_csl_video(video_dir, split, video_id):
    for letter in CE_CSL_LETTER_DIRS:
        path = Path(video_dir) / split / letter / f"{video_id}.mp4"
        if path.exists():
            return path
    return None


def process_frame_sequence(landmarker, frame_dir, save_path):
    jpgs = sorted(Path(frame_dir).glob("*.jpg"))
    if not jpgs:
        return 0, False

    frames = []
    for jpg_path in jpgs:
        frame_bgr = cv2.imread(str(jpg_path))
        if frame_bgr is None:
            continue
        kp = extract_keypoints(landmarker, frame_bgr)
        frames.append(kp)

    if len(frames) == 0:
        return 0, False

    sequence = np.stack(frames, axis=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), sequence)
    return len(frames), True


# ─── Dataset runners ───────────────────────────────────────────────────────

def run_ce_csl(args, landmarker):
    label_map = load_ce_csl_labels(args.label_dir, args.split)
    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    total, processed, skipped, errors = len(label_map), 0, 0, 0
    start = time.time()

    for video_id, gloss in label_map.items():
        if args.max_videos and processed >= args.max_videos:
            break

        save_path = out_dir / f"{video_id}.npy"
        if save_path.exists():
            skipped += 1
            continue

        video_path = find_ce_csl_video(args.video_dir, args.split, video_id)
        if video_path is None:
            if video_id != "train-01418":
                print(f"  [WARN] {video_id}: video not found")
            errors += 1
            continue

        n_frames, ok = process_video(landmarker, video_path, save_path)
        if ok:
            processed += 1
        else:
            errors += 1

        elapsed = time.time() - start
        eta = (elapsed / max(processed, 1)) * (min(total, args.max_videos or total) - processed - skipped)
        print(f"  [{processed}/{total}] {video_id}: {n_frames}f | "
              f"gloss: {gloss[:30]} | ETA: {eta / 60:.1f}min")

    print(f"\n{split} done: {processed} processed, {skipped} skipped, {errors} errors")


def run_slr(args, landmarker):
    """SLR isolated word processing.
    Frame sequences: {video_dir}/{word_idx}/{session}/000001.jpg ... 000067.jpg
    Train/dev: 90/10 stratified per word.
    Each session is one video sample.
    """
    word_map = {}
    with open(args.slr_dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                word_map[parts[0]] = parts[1]

    video_map = defaultdict(list)
    vd = Path(args.video_dir)
    for word_dir in vd.iterdir():
        if not word_dir.is_dir():
            continue
        for session_dir in word_dir.iterdir():
            if not session_dir.is_dir():
                continue
            jpgs = list(session_dir.glob("*.jpg"))
            if len(jpgs) >= 3:
                video_map[word_dir.name].append(str(session_dir))

    if not video_map:
        print("ERROR: No videos found. Extract SLR_videos_frm_cut*.zip first.")
        return

    total_videos = sum(len(v) for v in video_map.values())
    print(f"SLR: {total_videos} videos, {len(video_map)} words")

    train_list, dev_list = [], []
    for word_idx, videos in sorted(video_map.items()):
        n_dev = max(1, int(len(videos) * 0.1))
        word = word_map.get(word_idx, word_idx)
        for i, vp in enumerate(videos):
            if i < n_dev:
                dev_list.append((vp, word_idx, word))
            else:
                train_list.append((vp, word_idx, word))

    print(f"  Train: {len(train_list)}, Dev: {len(dev_list)}")

    for split_name, video_list in [("train", train_list), ("dev", dev_list)]:
        out_dir = Path(args.output_dir) / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        total = len(video_list)
        processed, errors = 0, 0

        for i, (vpath, word_idx, word) in enumerate(video_list):
            if args.max_videos and processed >= args.max_videos:
                break

            video_id = f"{word_idx}_{Path(vpath).stem}"
            save_path = out_dir / f"{video_id}.npy"
            if save_path.exists():
                continue

            n_frames, ok = process_frame_sequence(landmarker, Path(vpath), save_path)
            if ok:
                processed += 1
            else:
                errors += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                eta = (elapsed / max(processed, 1)) * (total - processed)
                print(f"  [{split_name}] {processed}/{total} | word: {word} | "
                      f"ETA: {eta / 60:.1f}min")

        print(f"  {split_name} done: {processed} processed, {errors} errors")

    label_path = Path(args.output_dir) / "labels.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(word_map, f, ensure_ascii=False, indent=2)
    print(f"Labels saved to {label_path}")


def run_csl_daily(args, landmarker):
    video_info, video_split = load_csl_daily_labels(args.label_dir)

    if args.split == "all":
        splits_to_do = sorted(set(video_split.values()))
    else:
        splits_to_do = [args.split]

    frame_dir = Path(args.video_dir)

    for split in splits_to_do:
        out_dir = Path(args.output_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)

        split_videos = [vid for vid, s in video_split.items() if s == split]
        total = len(split_videos)
        processed, errors = 0, 0
        start = time.time()

        for video_id in split_videos:
            if args.max_videos and processed >= args.max_videos:
                break

            save_path = out_dir / f"{video_id}.npy"
            if save_path.exists():
                continue

            vid_frame_dir = frame_dir / video_id
            if not vid_frame_dir.is_dir():
                errors += 1
                continue

            n_frames, ok = process_frame_sequence(landmarker, vid_frame_dir, save_path)
            if ok:
                processed += 1
            else:
                errors += 1

            if (processed + errors) % 200 == 0:
                elapsed = time.time() - start
                done = processed + errors
                eta = (elapsed / max(done, 1)) * (total - done)
                print(f"  [{split}] {done}/{total} ({processed} ok, {errors} err) | "
                      f"ETA: {eta / 60:.1f}min")

        print(f"  {split} done: {processed} processed, {errors} errors")

    label_path = Path(args.output_dir) / "labels.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump({vid: glosses for vid, glosses in video_info.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"Labels saved to {label_path}")


# ─── Main entry ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Multi-dataset keypoint preprocessing")
    ap.add_argument("--dataset", type=str, default="ce-csl",
                    choices=["ce-csl", "slr", "csl-daily"])
    ap.add_argument("--split", type=str, default="all")
    ap.add_argument("--max_videos", type=int, default=None)
    ap.add_argument("--video-dir", type=str, default=None)
    ap.add_argument("--label-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--slr-dict-path", type=str, default=None)
    args = ap.parse_args()

    BASE = Path("D:/red star project")

    if args.video_dir is None:
        if args.dataset == "ce-csl":
            args.video_dir = str(BASE / "CE-CSL/CE-CSL/video")
        elif args.dataset == "slr":
            args.video_dir = str(BASE / "SLR_Dataset/【孤立词】SLR_dataset/[fram_cut]slr500_rgb")
        elif args.dataset == "csl-daily":
            args.video_dir = "E:/CSL-Daily/CSL-Daily/sentence/frames_512x512"

    if args.label_dir is None:
        if args.dataset == "ce-csl":
            args.label_dir = str(BASE / "CE-CSL/CE-CSL/label")
        elif args.dataset == "csl-daily":
            args.label_dir = "E:/CSL-Daily/CSL-Daily/sentence_label"

    if args.output_dir is None:
        if args.dataset == "ce-csl":
            args.output_dir = str(BASE / "CE-CSL/CE-CSL/keypoints")
        elif args.dataset == "slr":
            args.output_dir = str(BASE / "SLR_Dataset/keypoints")
        elif args.dataset == "csl-daily":
            args.output_dir = "E:/CSL-Daily/CSL-Daily/keypoints"

    if args.model_path is None:
        args.model_path = str(BASE / "models/hand_landmarker.task")

    if args.slr_dict_path is None:
        args.slr_dict_path = str(BASE / "SLR_Dataset/【孤立词】SLR_dataset/dictionary.txt")

    print(f"Dataset: {args.dataset}  split: {args.split}")
    print(f"Video: {args.video_dir}")
    print(f"Output: {args.output_dir}")

    landmarker = create_landmarker(args.model_path)

    if args.dataset == "ce-csl":
        run_ce_csl(args, landmarker)
    elif args.dataset == "slr":
        run_slr(args, landmarker)
    elif args.dataset == "csl-daily":
        run_csl_daily(args, landmarker)

    landmarker.close()


if __name__ == "__main__":
    main()
